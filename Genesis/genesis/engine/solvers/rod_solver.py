# pylint: disable=no-value-for-parameter

import torch
import numpy as np
from math import pi
import gstaichi as ti
import gstaichi.math as tm
from typing import Iterable

import genesis as gs
from genesis.engine.boundaries import FloorBoundaryForRods as FloorBoundary
from genesis.engine.entities.rod_entity import RodEntity
from genesis.engine.states.solvers import RODSolverState
from genesis.utils.misc import ti_field_to_torch

from .base_solver import Solver

EPS = 1e-14


@ti.func
def get_perpendicular_vector(vector):
    """
    Returns a *unit* vector perpendicular to the input vector.
    """
    # Pick axis least aligned with vector
    abs_vector = ti.abs(vector)

    a = ti.Vector([0.0, 0.0, 0.0])
    if abs_vector.x <= abs_vector.y and abs_vector.x <= abs_vector.z:
        a = ti.Vector([1.0, 0.0, 0.0])
    elif abs_vector.y <= abs_vector.z:
        a = ti.Vector([0.0, 1.0, 0.0])
    else:
        a = ti.Vector([0.0, 0.0, 1.0])
    return tm.cross(vector, a).normalized()

@ti.func
def parallel_transport_normalized(t0, t1, v):
    """
    Transport vector :math:`v` from edge with tangent vector :math:`e0` to edge with tangent
    vector :math:`e1` (edge tangent vectors are normalized)
    """
    sin_theta_axis = tm.cross(t0, t1)
    cos_theta = tm.dot(t0, t1)
    den = 1 + cos_theta # denominator

    vprime = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(den) < EPS:
        vprime = v

    elif ti.abs(t0.x - t1.x) < EPS and ti.abs(t0.y - t1.y) < EPS and ti.abs(t0.z - t1.z) < EPS:
        vprime = v

    else:
        vprime = cos_theta * v + tm.cross(sin_theta_axis, v) + (tm.dot(sin_theta_axis, v) / den) * sin_theta_axis
    return vprime

@ti.func
def curvature_binormal(e0, e1):
    """
    Compute the curvature binormal for a vertex between two edges with tangents
    :math:`e0` and :math:`e1`, respectively (edge tangent vectors *not* necessarily normalized)
    """
    return 2.0 * tm.cross(e0, e1) / (tm.length(e0) * tm.length(e1) + tm.dot(e0, e1))

@ti.func
def get_updated_material_frame(prev_d3, d3, ref_d1, ref_d2, theta):
    """
    Parallel transport the reference frame vectors :math:`ref_d1` and :math:`ref_d2` from
    the previous edge to the new tangent vector :math:`d3` to get the updated reference frame.
    Then, rotate them by the twist angle :math:`theta` to get the updated material frame.
    """
    ref_d1 = parallel_transport_normalized(prev_d3, d3, ref_d1)
    ref_d2 = parallel_transport_normalized(prev_d3, d3, ref_d2)
    d1 = ti.cos(theta) * ref_d1 + ti.sin(theta) * ref_d2
    d2 = -ti.sin(theta) * ref_d1 + ti.cos(theta) * ref_d2
    return d1, d2, ref_d1, ref_d2

@ti.func
def get_angle(a, vec1, vec2):
    """
    Get the signed angle from :math:`vec1` to :math:`vec2` around axis :math:`a`; 
    ccw angles are positive. Assumes all vectors are *normalized* and *perpendicular* to :math:`a`
    Output in the range :math:`[-pi, pi]`
    """
    s = ti.max(-1.0, ti.min(1.0, tm.cross(vec1, vec2).dot(a)))
    c = ti.max(-1.0, ti.min(1.0, tm.dot(vec1, vec2)))
    return tm.atan2(s, c)

@ti.func
def get_updated_reference_twist(ref_d1_im1, ref_d1, d3_im1, d3):
    """
    Get the reference twist angle for the current edge based on the previous edge's
    reference director :math:`ref_d1_im1`, the current edge's reference director 
    :math:`ref_d1`, and the previous and current edge's tangent vectors :math:`d3_im1` 
    and :math:`d3`. Assumes all vectors are *normalized*.
    """
    # Finite rotation angle needed to take the parallel transported copy
    # of the previous edge's reference director to the current edge's
    # reference director.
    vec1 = parallel_transport_normalized(d3_im1, d3, ref_d1_im1)
    vec2 = ref_d1
    reference_twist = get_angle(d3, vec1, vec2)
    return reference_twist

@ti.func
def quat_rotate(q: tm.vec4, v: tm.vec3) -> tm.vec3:
    """
    Rotate vector `v` by quaternion `q`.
    """
    qvec = ti.Vector([q[1], q[2], q[3]])
    uv = tm.cross(qvec, v)
    uuv = tm.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


@ti.data_oriented
class RodSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._adjacent_gap = options.adjacent_gap
        self._damping = options.damping
        self._angular_damping = options.angular_damping
        self._n_pbd_iters = options.n_pbd_iters
        self._max_collision_grad_norm = 0.1

        # boundary
        self.setup_boundary()

        # lazy initialization
        self._constraints_initialized = False

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif isinstance(shape, (list, tuple)):
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_rod_fields(self):
        # rod information (static)
        struct_rod_info = ti.types.struct(
            # material properties
            use_inextensible=gs.ti_bool,
            stretching_stiffness=gs.ti_float,
            bending_stiffness=gs.ti_float,
            twisting_stiffness=gs.ti_float,
            plastic_yield=gs.ti_float,
            plastic_creep=gs.ti_float,

            # indices
            first_vert_idx=gs.ti_int,           # index of the first vertex of this rod
            first_edge_idx=gs.ti_int,           # index of the first edge of this rod
            first_internal_vert_idx=gs.ti_int,  # index of the first internal vertex of this rod
            n_verts=gs.ti_int,                  # number of vertices in this rod

            # is loop
            is_loop=gs.ti_bool,
        )

        # rod energy (w/o time dimension)
        struct_rod_energy = ti.types.struct(
            stretching_energy=gs.ti_float,
            bending_energy=gs.ti_float,
            twisting_energy=gs.ti_float,
        )

        self.rods_info = struct_rod_info.field(
            shape=self._n_rods, layout=ti.Layout.SOA
        )

        self.rods_energy = struct_rod_energy.field(
            shape=self._batch_shape(self._n_rods),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        # keep track of gradients for time-stepping
        self.gradients = ti.field(
            dtype=gs.ti_float, needs_grad=False,
            shape=self._batch_shape(self._n_dofs)
        )

    def init_vertex_fields(self):
        # vertex information (static)
        struct_vertex_info = ti.types.struct(
            mass=gs.ti_float,
            radius=gs.ti_float,
            # vert_rest=gs.ti_vec3,
            mu_s=gs.ti_float,
            mu_k=gs.ti_float,
            restitution=gs.ti_float,  # coefficient of restitution for self-collision
            rod_idx=gs.ti_int,        # index of the rod this vertex belongs to
        )

        # vertex state (dynamic)
        struct_vertex_state = ti.types.struct(
            vert=gs.ti_vec3,        # current position
            vel=gs.ti_vec3,
        )

        # vertex force (w/o time dimension)
        struct_vertex_force = ti.types.struct(
            f_s=gs.ti_vec3,             # stretching force
            f_b=gs.ti_vec3,             # bending force
            f_t=gs.ti_vec3,             # twisting force
        )

        struct_vertex_state_ng = ti.types.struct(
            fixed=gs.ti_bool,           # is the vertex fixed
            is_kinematic=gs.ti_bool,    # is the vertex kinematic
            is_collided=gs.ti_bool,
        )

        self.vertices_info = struct_vertex_info.field(
            shape=self._n_vertices, layout=ti.Layout.SOA
        )

        self.vertices = struct_vertex_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=True,
            layout=ti.Layout.SOA
        )

        self.vertices_force = struct_vertex_force.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        self.vertices_ng = struct_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        # for visualization
        self.vertices_render = ti.Vector.field(
            3, dtype=gs.ti_float, needs_grad=False,
            shape=self._batch_shape(self._n_vertices)
        )

    def init_edge_fields(self):
        # edge information (static)
        struct_edge_info = ti.types.struct(
            edge_rest=gs.ti_vec3,
            length_rest=gs.ti_float,
            d1_rest=gs.ti_vec3,         # material frame direction 1 in rest state
            d2_rest=gs.ti_vec3,         # material frame direction 2 in rest state
            d3_rest=gs.ti_vec3,         # material frame direction 3 in rest state (tangent)
            vert_idx=gs.ti_int,         # index of the starting vertex of this edge
        )

        # edge state (dynamic)
        struct_edge_state = ti.types.struct(
            theta=gs.ti_float,      # twist angle
            omega=gs.ti_float,      # twist rate (angular velocity)
        )

        struct_edge_state_ng = ti.types.struct(
            edge=gs.ti_vec3,        # current edge vector
            length=gs.ti_float,     # current edge length
            d1=gs.ti_vec3,          # material frame direction 1
            d2=gs.ti_vec3,          # material frame direction 2
            d3=gs.ti_vec3,          # material frame direction 3 (tangent)
            d1_ref=gs.ti_vec3,      # reference material frame direction 1
            d2_ref=gs.ti_vec3,      # reference material frame direction 2
        )

        self.edges_info = struct_edge_info.field(
            shape=self._n_edges, layout=ti.Layout.SOA
        )

        self.edges = struct_edge_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_edges)),
            needs_grad=True,
            layout=ti.Layout.SOA
        )

        self.edges_ng = struct_edge_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_edges)),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

    def init_internal_vertex_fields(self):
        # internal vertex information (static)
        struct_internal_vertex_info = ti.types.struct(
            twist_rest=gs.ti_float,     # rest twist
            edge_idx=gs.ti_int,         # index of the starting edge of this internal vertex
        )

        struct_internal_vertex_state_ng = ti.types.struct(
            kb=gs.ti_vec3,          # current curvature binormal
            twist=gs.ti_float,      # current twist
            kappa_rest=gs.ti_vec2,      # rest curvature,
        )

        self.internal_vertices_info = struct_internal_vertex_info.field(
            shape=self._n_internal_vertices, layout=ti.Layout.SOA
        )

        self.internal_vertices_ng = struct_internal_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_internal_vertices)),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

    def init_constraints(self):
        # NOTE: call this after call `_kernel_add_rods`
        valid_edge_pairs = list()
        for i in range(self._n_vertices):
            for j in range(i + 1, self._n_vertices):
                rod_id_i = self.vertices_info[i].rod_idx
                local_id_i = i - self.rods_info[rod_id_i].first_vert_idx
                rod_id_j = self.vertices_info[j].rod_idx
                local_id_j = j - self.rods_info[rod_id_j].first_vert_idx

                # filtering
                # 1. ensure i and j can actually start an edge
                is_loop_i = self.rods_info[rod_id_i].is_loop
                is_loop_j = self.rods_info[rod_id_j].is_loop

                if not is_loop_i and local_id_i >= self.rods_info[rod_id_i].n_verts - 1:
                    continue
                if not is_loop_j and local_id_j >= self.rods_info[rod_id_j].n_verts - 1:
                    continue

                # 2. ignore adjacent edges on the same rod
                if rod_id_i == rod_id_j:
                    if is_loop_i:
                        n_verts_in_rod = self.rods_info[rod_id_i].n_verts
                        dist_forward = local_id_j - local_id_i
                        dist_backward = (local_id_i + n_verts_in_rod) - local_id_j

                        if dist_forward < self._adjacent_gap + 1 or dist_backward < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the loop.
                    else:
                        if abs(local_id_j - local_id_i) < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the chain.

                valid_edge_pairs.append((i, j))

        valid_edge_pairs = np.array(valid_edge_pairs, dtype=gs.np_int)
        self._n_valid_edge_pairs = valid_edge_pairs.shape[0]

        # constraint for rod-rod collision
        struct_rr_info = ti.types.struct(
            valid_pair=ti.types.vector(2, gs.ti_int),
        )

        struct_rr_state = ti.types.struct(
            normal=gs.ti_vec3,
            penetration=gs.ti_float,
        )

        self.rr_constraint_info = struct_rr_info.field(
            shape=self._n_valid_edge_pairs, layout=ti.Layout.SOA
        )

        self.rr_constraints = struct_rr_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_valid_edge_pairs)),
            needs_grad=True,
            layout=ti.Layout.AOS
        )

        self.rr_constraint_info.valid_pair.from_numpy(valid_edge_pairs)

        self._constraints_initialized = True

    def register_gripper_geom_indices(self, geom_indices: Iterable[int]=()):
        """
        Register the geometry indices of the gripper for collision handling.
        Needs to be called before building the scene.
        """
        geom_indices = np.asarray(geom_indices, dtype=gs.np_int)
        field_shape = max(geom_indices.shape[0], 1)
        self.geom_indices = ti.field(
            dtype=gs.ti_int, needs_grad=False, shape=field_shape
        )
        if geom_indices.shape[0] > 0:
            self.geom_indices.from_numpy(geom_indices)
        else:
            self.geom_indices[0] = -1
        self._n_geom_indices = geom_indices.shape[0]
        gs.logger.info(f"Registered {geom_indices.shape[0]} gripper geometries for rod collision handling.")
        gs.logger.info(f"Geom indices: {geom_indices}")

    @ti.func
    def _func_is_geom_idx_registered(self, i_g: ti.i32):
        registered = False
        ti.loop_config(serialize=True)
        for i in range(self._n_geom_indices):
            if self.geom_indices[i] == i_g:
                registered = True
                break
        return registered

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.vertices.grad.fill(0)
        self.edges.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._n_rods = self.n_rods
        self._n_vertices = self.n_vertices
        self._n_edges = self.n_edges
        self._n_internal_vertices = self.n_internal_vertices
        self._n_dofs = self.n_dofs

        # rendering
        self.envs_offset = ti.Vector.field(3, dtype=ti.f32, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(np.float32))

        if self.is_active():
            self.init_rod_fields()
            self.init_vertex_fields()
            self.init_edge_fields()
            self.init_internal_vertex_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_constraints()

    def add_entity(self, idx, material, morph, surface, visualize_twist):

        # create entity
        entity = RodEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            rod_idx=self.n_rods,
            v_start=self.n_vertices,
            e_start=self.n_edges,
            iv_start=self.n_internal_vertices,
            visualize_twist=visualize_twist,
        )

        self._entities.append(entity)
        return entity

    def is_active(self):
        return self._n_vertices > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ logging --------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def get_rod_length(self, f: ti.i32, i_r: ti.i32, length: ti.types.ndarray()):
        n_verts = self.rods_info[i_r].n_verts
        first_edge_idx = self.rods_info[i_r].first_edge_idx
        for i_e, i_b in ti.ndrange(n_verts - 1, self._B):
            edge_idx = first_edge_idx + i_e
            length[i_b] += self.edges_ng[f, edge_idx, i_b].length

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def _func_clear_energy(self):
        for i_r, i_b in ti.ndrange(self._n_rods, self._B):
            self.rods_energy[i_r, i_b].stretching_energy = 0.0
            self.rods_energy[i_r, i_b].bending_energy = 0.0
            self.rods_energy[i_r, i_b].twisting_energy = 0.0

    @ti.func
    def _func_clear_force(self):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices_force[i_v, i_b].f_s = ti.Vector.zero(gs.ti_float, 3)
            self.vertices_force[i_v, i_b].f_b = ti.Vector.zero(gs.ti_float, 3)
            self.vertices_force[i_v, i_b].f_t = ti.Vector.zero(gs.ti_float, 3)

    @ti.func
    def _func_clear_gradients(self):
        self.gradients.fill(0.0)

    @ti.kernel
    def update_centerline_positions(self, f: ti.i32):      # Differential    # FIXME: check if correct
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            if not self.vertices_ng[f, i_v, i_b].fixed:
                # self.vertices[f + 1, i_v, i_b].vert += self.vertices[f + 1, i_v, i_b].vel * self.substep_dt
                self.vertices[f + 1, i_v, i_b].vert = (
                    self.vertices[f + 1, i_v, i_b].vel * self.substep_dt + self.vertices[f, i_v, i_b].vert
                )

    @ti.kernel
    def update_centerline_velocities(self, f: ti.i32):       # Differential
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            mass = self.vertices_info[i_v].mass
            if not self.vertices_ng[f, i_v, i_b].fixed:
                gradient = ti.Vector([
                    self.gradients[3 * i_v + 0, i_b],
                    self.gradients[3 * i_v + 1, i_b],
                    self.gradients[3 * i_v + 2, i_b],
                ])
                self.vertices[f + 1, i_v, i_b].vel -= gradient / mass * self.substep_dt

                # apply damping if enabled
                self.vertices[f + 1, i_v, i_b].vel *= ti.exp(-self.substep_dt * self.damping)
                # self.vertices[f, i_v, i_b].vel *= (1.0 - self.damping)
                # add gravity (avoiding damping on gravity)
                self.vertices[f + 1, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @ti.kernel
    def update_angular_velocities(self, f: ti.i32):      # Differential
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            if not self.vertices_ng[f, v_s, i_b].fixed or not self.vertices_ng[f, v_e, i_b].fixed:
                theta_dof_idx = 3 * self._n_vertices + i_e
                gradient = self.gradients[theta_dof_idx, i_b]
                inertia = 1.0
                self.edges[f + 1, i_e, i_b].omega -= gradient / inertia * self.substep_dt
                self.edges[f + 1, i_e, i_b].omega *= ti.exp(-self.substep_dt * self.angular_damping)

    @ti.kernel
    def update_centerline_edges(self, f: ti.i32):    # Differential
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            self.edges_ng[f + 1, i_e, i_b].edge = self.vertices[f + 1, v_e, i_b].vert - self.vertices[f + 1, v_s, i_b].vert
            self.edges_ng[f + 1, i_e, i_b].length = tm.length(self.edges_ng[f + 1, i_e, i_b].edge)

    @ti.kernel
    def update_frame_thetas(self, f: ti.i32):      # Differential
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            if not self.vertices_ng[f, v_s, i_b].fixed or not self.vertices_ng[f, v_e, i_b].fixed:
                # self.edges[f + 1, i_e, i_b].theta -= self.gradients[3 * self._n_vertices + i_e, i_b] * self.substep_dt
                self.edges[f + 1, i_e, i_b].theta = (
                    self.edges[f + 1, i_e, i_b].omega * self.substep_dt + self.edges[f, i_e, i_b].theta
                )

    @ti.kernel
    def update_material_states(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges_ng[f + 1, i_e, i_b].d3 = self.edges_ng[f + 1, i_e, i_b].edge.normalized()

            d1, d2, d1_ref, d2_ref = get_updated_material_frame(
                self.edges_ng[f, i_e, i_b].d3,         # prev d3
                self.edges_ng[f + 1, i_e, i_b].d3,     # curr d3
                self.edges_ng[f, i_e, i_b].d1_ref,
                self.edges_ng[f, i_e, i_b].d2_ref,
                self.edges[f + 1, i_e, i_b].theta,
            )
            self.edges_ng[f + 1, i_e, i_b].d1 = d1
            self.edges_ng[f + 1, i_e, i_b].d2 = d2
            self.edges_ng[f + 1, i_e, i_b].d1_ref = d1_ref
            self.edges_ng[f + 1, i_e, i_b].d2_ref = d2_ref

        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)

            self.internal_vertices_ng[f + 1, i_iv, i_b].kb = curvature_binormal(
                self.edges_ng[f + 1, e_s, i_b].d3, self.edges_ng[f + 1, e_e, i_b].d3
            )
            twist_ref = get_updated_reference_twist(
                self.edges_ng[f + 1, e_s, i_b].d1_ref, self.edges_ng[f + 1, e_e, i_b].d1_ref,
                self.edges_ng[f + 1, e_s, i_b].d3, self.edges_ng[f + 1, e_e, i_b].d3
            )
            self.internal_vertices_ng[f + 1, i_iv, i_b].twist = self.edges[f + 1, e_e, i_b].theta - self.edges[f + 1, e_s, i_b].theta + twist_ref

    @ti.kernel
    def update_velocities_after_projection(self, f: ti.i32):   # Differential
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            if not self.vertices_ng[f, i_v, i_b].fixed:
                self.vertices[f + 1, i_v, i_b].vel = (self.vertices[f + 1, i_v, i_b].vert - self.vertices[f, i_v, i_b].vert) / self.substep_dt

    @ti.kernel
    def transfer_fixed_states(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices_ng[f + 1, i_v, i_b].fixed = self.vertices_ng[f, i_v, i_b].fixed

    @ti.kernel
    def init_pos_and_vel(self, f: ti.i32):  # Differential
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices[f + 1, i_v, i_b].vert = self.vertices[f, i_v, i_b].vert
            self.vertices[f + 1, i_v, i_b].vel = self.vertices[f, i_v, i_b].vel

    @ti.kernel 
    def clear_collision_record(self):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices_ng[0, i_v, i_b].is_collided = False

    @ti.kernel
    def init_theta_and_omega(self, f: ti.i32):     # Differential
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges[f + 1, i_e, i_b].theta = self.edges[f, i_e, i_b].theta
            self.edges[f + 1, i_e, i_b].omega = self.edges[f, i_e, i_b].omega

    @ti.kernel
    def compute_stretching_energy(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check stretching enabled
            K = self.rods_info[rod_id].stretching_stiffness
            if K > 0.:
                r = (self.vertices_info[v_s].radius + self.vertices_info[v_e].radius) * 0.5
                a, b = r, r
                A = pi * a * b  # cross-sectional area

                strain_i = (self.edges_ng[f, i_e, i_b].length / self.edges_info[i_e].length_rest) - 1.0

                self.rods_energy[rod_id, i_b].stretching_energy += 0.5 * K * A * ti.pow(strain_i, 2) * self.edges_info[i_e].length_rest

                # -------------------------------- gradients --------------------------------

                gradient_magnitude = K * A * strain_i

                gradient_dx_i   = - gradient_magnitude * self.edges_ng[f, i_e, i_b].d3
                gradient_dx_ip1 =   gradient_magnitude * self.edges_ng[f, i_e, i_b].d3

                for k in range(3):
                    ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_ip1[k])

                    ti.atomic_add(self.vertices_force[v_s, i_b].f_s[k], -gradient_dx_i[k])
                    ti.atomic_add(self.vertices_force[v_e, i_b].f_s[k], -gradient_dx_ip1[k])

    @ti.kernel
    def compute_bending_energy(self, f: ti.i32):
        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check bending enabled
            E = self.rods_info[rod_id].bending_stiffness
            if E > 0.:
                r = self.vertices_info[v_m].radius
                a, b = r, r
                A = pi * a * b  # cross-sectional area
                B11 = E * A * ti.pow(a, 2) / 4.0
                B22 = E * A * ti.pow(b, 2) / 4.0

                kb = self.internal_vertices_ng[f, i_iv, i_b].kb
                l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5

                kappa1_i =   0.5 * tm.dot(kb, self.edges_ng[f, e_s, i_b].d2 + self.edges_ng[f, e_e, i_b].d2)
                kappa2_i = - 0.5 * tm.dot(kb, self.edges_ng[f, e_s, i_b].d1 + self.edges_ng[f, e_e, i_b].d1)

                # bending plasticity
                kappa1_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[0]
                kappa2_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[1]
                curr_kappa = ti.Vector([kappa1_i, kappa2_i])
                rest_kappa = ti.Vector([kappa1_rest_i, kappa2_rest_i])

                elastic_kappa = curr_kappa - rest_kappa
                elastic_kappa_norm = tm.length(elastic_kappa)

                yield_thres = self.rods_info[rod_id].plastic_yield
                creep_rate = self.rods_info[rod_id].plastic_creep

                yield_amount = elastic_kappa_norm - yield_thres
                if yield_amount > 0.:
                    # delta_rest_kappa = self.substep_dt * creep_rate * (yield_amount / elastic_kappa_norm) * elastic_kappa
                    delta_rest_kappa = creep_rate * (yield_amount / elastic_kappa_norm) * elastic_kappa
                    self.internal_vertices_ng[f + 1, i_iv, i_b].kappa_rest = (
                        delta_rest_kappa + self.internal_vertices_ng[f, i_iv, i_b].kappa_rest
                    )
                    # print(f"Rod {rod_id}, iv {i_iv}, yield_amount: {yield_amount}")
                else:
                    # f -> f+1
                    self.internal_vertices_ng[f + 1, i_iv, i_b].kappa_rest = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest

                kappa1_rest_i = self.internal_vertices_ng[f + 1, i_iv, i_b].kappa_rest[0]
                kappa2_rest_i = self.internal_vertices_ng[f + 1, i_iv, i_b].kappa_rest[1]

                self.rods_energy[rod_id, i_b].bending_energy += 0.5 * (
                    B11 * ti.pow(kappa1_i - kappa1_rest_i, 2) +
                    B22 * ti.pow(kappa2_i - kappa2_rest_i, 2)
                ) / l_i

                # -------------------------------- gradients --------------------------------

                gradient_kappa1_i_x_i = ti.Vector.zero(dt=gs.ti_float, n=9)
                gradient_kappa2_i_x_i = ti.Vector.zero(dt=gs.ti_float, n=9)

                chi = 1. + tm.dot(self.edges_ng[f, e_s, i_b].d3, self.edges_ng[f, e_e, i_b].d3)
                d1_tilde = (self.edges_ng[f, e_s, i_b].d1 + self.edges_ng[f, e_e, i_b].d1) / chi
                d2_tilde = (self.edges_ng[f, e_s, i_b].d2 + self.edges_ng[f, e_e, i_b].d2) / chi
                d3_tilde = (self.edges_ng[f, e_s, i_b].d3 + self.edges_ng[f, e_e, i_b].d3) / chi

                dkappa1_i_de_im1 = tm.cross(d2_tilde, -self.edges_ng[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                    kappa1_i * d3_tilde / self.edges_info[e_s].length_rest
                dkappa1_i_de_i = tm.cross(d2_tilde, self.edges_ng[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                    kappa1_i * d3_tilde / self.edges_info[e_e].length_rest
                dkappa2_i_de_im1 = tm.cross(d1_tilde, self.edges_ng[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                    kappa2_i * d3_tilde / self.edges_info[e_s].length_rest
                dkappa2_i_de_i = tm.cross(d1_tilde, -self.edges_ng[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                    kappa2_i * d3_tilde / self.edges_info[e_e].length_rest

                gradient_kappa1_i_x_i[0:3] = dkappa1_i_de_im1 * (- 1.0)
                gradient_kappa1_i_x_i[3:6] = dkappa1_i_de_im1 * (  1.0) + dkappa1_i_de_i * (- 1.0)
                gradient_kappa1_i_x_i[6:9] = dkappa1_i_de_i   * (  1.0)
                gradient_kappa2_i_x_i[0:3] = dkappa2_i_de_im1 * (- 1.0)
                gradient_kappa2_i_x_i[3:6] = dkappa2_i_de_im1 * (  1.0) + dkappa2_i_de_i * (- 1.0)
                gradient_kappa2_i_x_i[6:9] = dkappa2_i_de_i   * (  1.0)

                gradient_dx_i = (
                    B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_x_i + \
                    B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_x_i
                ) / l_i
                for k in range(3):
                    ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    ti.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                    ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                    ti.atomic_add(self.vertices_force[v_s, i_b].f_b[k], -gradient_dx_i[k])
                    ti.atomic_add(self.vertices_force[v_m, i_b].f_b[k], -gradient_dx_i[k + 3])
                    ti.atomic_add(self.vertices_force[v_e, i_b].f_b[k], -gradient_dx_i[k + 6])

                gradient_kappa1_i_theta_i = - ti.Vector([
                    tm.dot(kb, self.edges_ng[f, e_s, i_b].d1) * 0.5,
                    tm.dot(kb, self.edges_ng[f, e_e, i_b].d1) * 0.5
                ])
                gradient_kappa2_i_theta_i = - ti.Vector([
                    tm.dot(kb, self.edges_ng[f, e_s, i_b].d2) * 0.5,
                    tm.dot(kb, self.edges_ng[f, e_e, i_b].d2) * 0.5
                ])

                gradient_dtheta_i = (
                    B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_theta_i + \
                    B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_theta_i
                ) / l_i
                theta_dof_s_idx = 3 * self._n_vertices + e_s
                theta_dof_e_idx = 3 * self._n_vertices + e_e
                ti.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
                ti.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @ti.kernel
    def compute_twisting_energy(self, f: ti.i32):
        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check twisting enabled
            G = self.rods_info[rod_id].twisting_stiffness
            if G > 0.:
                r = self.vertices_info[v_m].radius
                a, b = r, r
                A = pi * a * b  # cross-sectional area
                beta = G * A * (ti.pow(a, 2) + ti.pow(b, 2)) / 4.0

                kb = self.internal_vertices_ng[f, i_iv, i_b].kb
                l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5
                m_i = self.internal_vertices_ng[f, i_iv, i_b].twist
                m_i_rest = self.internal_vertices_info[i_iv].twist_rest

                self.rods_energy[rod_id, i_b].twisting_energy += 0.5 * beta * ti.pow(m_i - m_i_rest, 2) / l_i

                # -------------------------------- gradients --------------------------------

                gradient_m_i_dx_i = ti.Vector.zero(dt=gs.ti_float, n=9)
                gradient_m_i_dx_i[0:3] = - kb / (2.0 * self.edges_ng[f, e_s, i_b].length)
                gradient_m_i_dx_i[3:6] =   kb / (2.0 * self.edges_ng[f, e_s, i_b].length) - kb / (2.0 * self.edges_ng[f, e_e, i_b].length)
                gradient_m_i_dx_i[6:9] =   kb / (2.0 * self.edges_ng[f, e_e, i_b].length)
                gradient_dx_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dx_i
                for k in range(3):
                    ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                    ti.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                    ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                    ti.atomic_add(self.vertices_force[v_s, i_b].f_t[k], -gradient_dx_i[k])
                    ti.atomic_add(self.vertices_force[v_m, i_b].f_t[k], -gradient_dx_i[k + 3])
                    ti.atomic_add(self.vertices_force[v_e, i_b].f_t[k], -gradient_dx_i[k + 6])

                gradient_m_i_dtheta_i = ti.Vector([-1.0, 1.0])
                gradient_dtheta_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dtheta_i
                theta_dof_s_idx = 3 * self._n_vertices + e_s
                theta_dof_e_idx = 3 * self._n_vertices + e_e
                ti.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
                ti.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @ti.kernel
    def clear_energy_and_gradients(self):
        # clear energy and gradients
        self._func_clear_force()
        self._func_clear_energy()
        self._func_clear_gradients()

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active():
            self.init_pos_and_vel(f)
            self.init_theta_and_omega(f)
            self.clear_energy_and_gradients()
            self.compute_stretching_energy(f)
            self.compute_bending_energy(f)
            self.compute_twisting_energy(f)
            self.update_centerline_velocities(f)
            self.update_angular_velocities(f)
            if f == 0:
                self.clear_collision_record()

    def substep_pre_coupling_grad(self, f):
        if self.is_active():
            self.update_angular_velocities.grad(f)
            self.update_centerline_velocities.grad(f)
            self.compute_twisting_energy.grad(f)
            self.compute_bending_energy.grad(f)
            self.compute_stretching_energy.grad(f)
            self.clear_energy_and_gradients.grad()
            self.init_theta_and_omega.grad(f)
            self.init_pos_and_vel.grad(f)

    def substep_post_coupling(self, f):
        if self.is_active():
            self.update_centerline_positions(f)
            self.update_frame_thetas(f)

            for i in range(self._n_pbd_iters):
                self._kernel_apply_inextensibility_constraints(f)
                # self._kernel_apply_rod_collision_constraints(f, i)
                self.collision_forward(f, i)
            self.update_centerline_edges(f)
            self.update_material_states(f)
            self.update_velocities_after_projection(f)
            # self._kernel_apply_rod_friction(f)
            self.friction_forward(f)

            self.transfer_fixed_states(f)   # f -> f+1

            # if f % 20 == 0:
            #     vert = self.vertices.vert.to_numpy()[f, :, 0]
            #     nan_mask = np.isnan(vert)
            #     if np.sum(nan_mask) > 0:
            #         gs.logger.warning(f"[Debug][RodSolver] NaN vertices: {len(np.where(nan_mask)[0])} / {vert.shape[0]}, id: {np.where(nan_mask)[0]}")
            #     length = np.zeros((self._B,), dtype=gs.np_float)
            #     self.get_rod_length(f, 0, length)
                # print(f"[Debug][RodSolver] rod length: {length}, v[0, 0]: {self.vertices[f, 0, 0].vert}")

    def substep_post_coupling_grad(self, f):
        if self.is_active():
            self.transfer_fixed_states.grad(f)

            self.friction_forward.grad(self, f)
            # self._kernel_apply_rod_friction.grad(f)
            self.update_velocities_after_projection.grad(f)
            self.update_material_states.grad(f)
            self.update_centerline_edges.grad(f)
            for i in range(self._n_pbd_iters)[::-1]:
                # self._kernel_apply_rod_collision_constraints.grad(f, i)
                self.collision_forward.grad(self, f, i)
                self._kernel_apply_inextensibility_constraints.grad(f)

            self.update_frame_thetas.grad(f)
            self.update_centerline_positions.grad(f)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices[target, i_v, i_b].vert = self.vertices[source, i_v, i_b].vert
            self.vertices[target, i_v, i_b].vel = self.vertices[source, i_v, i_b].vel

            self.vertices_ng[target, i_v, i_b].fixed = self.vertices_ng[source, i_v, i_b].fixed

        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges[target, i_e, i_b].theta = self.edges[source, i_e, i_b].theta
            self.edges[target, i_e, i_b].omega = self.edges[source, i_e, i_b].omega

            self.edges_ng[target, i_e, i_b].edge = self.edges_ng[source, i_e, i_b].edge
            self.edges_ng[target, i_e, i_b].length = self.edges_ng[source, i_e, i_b].length
            self.edges_ng[target, i_e, i_b].d1 = self.edges_ng[source, i_e, i_b].d1
            self.edges_ng[target, i_e, i_b].d2 = self.edges_ng[source, i_e, i_b].d2
            self.edges_ng[target, i_e, i_b].d3 = self.edges_ng[source, i_e, i_b].d3
            self.edges_ng[target, i_e, i_b].d1_ref = self.edges_ng[source, i_e, i_b].d1_ref
            self.edges_ng[target, i_e, i_b].d2_ref = self.edges_ng[source, i_e, i_b].d2_ref

        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            self.internal_vertices_ng[target, i_iv, i_b].kb = self.internal_vertices_ng[source, i_iv, i_b].kb
            self.internal_vertices_ng[target, i_iv, i_b].twist = self.internal_vertices_ng[source, i_iv, i_b].twist
            self.internal_vertices_ng[target, i_iv, i_b].kappa_rest = self.internal_vertices_ng[source, i_iv, i_b].kappa_rest

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices.grad[target, i_v, i_b].vert = self.vertices.grad[source, i_v, i_b].vert
            self.vertices.grad[target, i_v, i_b].vel = self.vertices.grad[source, i_v, i_b].vel

            self.vertices_ng[target, i_v, i_b].fixed = self.vertices_ng[source, i_v, i_b].fixed

        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges.grad[target, i_e, i_b].theta = self.edges.grad[source, i_e, i_b].theta
            self.edges.grad[target, i_e, i_b].omega = self.edges.grad[source, i_e, i_b].omega

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        # Zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
        for i_f, i_v, i_b in ti.ndrange(f, self._n_vertices, self._B):
            self.vertices.grad[i_f, i_v, i_b].vert = ti.Vector.zero(gs.ti_float, 3)
            self.vertices.grad[i_f, i_v, i_b].vel = ti.Vector.zero(gs.ti_float, 3)

        for i_f, i_e, i_b in ti.ndrange(f, self._n_edges, self._B):
            self.edges.grad[i_f, i_e, i_b].theta = 0.0
            self.edges.grad[i_f, i_e, i_b].omega = 0.0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active():
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)
        
            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.theta.grad is not None:
                state.theta.assert_contiguous()
                self.add_grad_from_theta(self._sim.cur_substep_local, state.theta.grad)

            if state.omega.grad is not None:
                state.omega.assert_contiguous()
                self.add_grad_from_omega(self._sim.cur_substep_local, state.omega.grad)

    @ti.kernel
    def add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        # pos_grad shape: [B, n_vertices, 3]
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for k in ti.static(range(3)):
                self.vertices.grad[f, i_v, i_b].vert[k] += pos_grad[i_b, i_v, k]

    @ti.kernel
    def add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        # vel_grad shape: [B, n_vertices, 3]
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for k in ti.static(range(3)):
                self.vertices.grad[f, i_v, i_b].vel[k] += vel_grad[i_b, i_v, k]

    @ti.kernel
    def add_grad_from_theta(self, f: ti.i32, theta_grad: ti.types.ndarray()):
        # theta_grad shape: [B, n_edges]
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges.grad[f, i_e, i_b].theta += theta_grad[i_b, i_e]

    @ti.kernel
    def add_grad_from_omega(self, f: ti.i32, omega_grad: ti.types.ndarray()):
        # omega_grad shape: [B, n_edges]
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges.grad[f, i_e, i_b].omega += omega_grad[i_b, i_e]

    def save_ckpt(self, ckpt_name):
        if self.is_active() and self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["vel"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["fixed"] = torch.zeros(
                    self._batch_shape((self.n_vertices,), first_dim=True), dtype=gs.tc_bool
                )
                self._ckpt[ckpt_name]["theta"] = torch.zeros(
                    self._batch_shape((self.n_edges,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["omega"] = torch.zeros(
                    self._batch_shape((self.n_edges,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d1"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d2"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d3"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d1_ref"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["d2_ref"] = torch.zeros(
                    self._batch_shape((self.n_edges, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["kb"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["twist"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices,), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["kappa_rest"] = torch.zeros(
                    self._batch_shape((self.n_internal_vertices, 2), first_dim=True), dtype=gs.tc_float
                )

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["fixed"],
                self._ckpt[ckpt_name]["theta"],
                self._ckpt[ckpt_name]["omega"],
                self._ckpt[ckpt_name]["d1"],
                self._ckpt[ckpt_name]["d2"],
                self._ckpt[ckpt_name]["d3"],
                self._ckpt[ckpt_name]["d1_ref"],
                self._ckpt[ckpt_name]["d2_ref"],
                self._ckpt[ckpt_name]["kb"],
                self._ckpt[ckpt_name]["twist"],
                self._ckpt[ckpt_name]["kappa_rest"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        if self.is_active():
            # restart from frame 0 in memory
            self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        if self.is_active():
            self.copy_frame(0, self._sim.substeps_local)
            self.copy_grad(0, self._sim.substeps_local)

        if self.is_active() and self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0, 
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["fixed"],
                self._ckpt[ckpt_name]["theta"],
                self._ckpt[ckpt_name]["omega"],
                self._ckpt[ckpt_name]["d1"],
                self._ckpt[ckpt_name]["d2"],
                self._ckpt[ckpt_name]["d3"],
                self._ckpt[ckpt_name]["d1_ref"],
                self._ckpt[ckpt_name]["d2_ref"],
                self._ckpt[ckpt_name]["kb"],
                self._ckpt[ckpt_name]["twist"],
                self._ckpt[ckpt_name]["kappa_rest"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active():
            self._kernel_set_state(
                f, state.pos, state.vel, state.fixed, 
                state.theta, state.omega,
                state.d1, state.d2, state.d3,
                state.d1_ref, state.d2_ref,
                state.kb, state.twist, state.kappa_rest,
            )

    def get_state(self, f):
        if self.is_active():
            state = RODSolverState(self._scene)
            self._kernel_get_state(
                f, state.pos, state.vel, state.fixed, 
                state.theta, state.omega,
                state.d1, state.d2, state.d3,
                state.d1_ref, state.d2_ref,
                state.kb, state.twist, state.kappa_rest,
            )
        else:
            state = None
        return state

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        vertices = self.vertices_render
        radii = self.vertices_info.radius

        return vertices, radii

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active():
            return None

        return ti_field_to_torch(self.elements_v_energy.force)

    @ti.kernel
    def _kernel_add_rods(
        self,
        rod_idx: ti.i32,
        is_loop: ti.u1,
        use_inextensible: ti.u1,
        stretching_stiffness: ti.f64,
        bending_stiffness: ti.f64,
        twisting_stiffness: ti.f64,
        plastic_yield: ti.f64,
        plastic_creep: ti.f64,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        n_verts: ti.i32,
    ):
        self.rods_info[rod_idx].use_inextensible = use_inextensible
        self.rods_info[rod_idx].stretching_stiffness = stretching_stiffness
        self.rods_info[rod_idx].bending_stiffness = bending_stiffness
        self.rods_info[rod_idx].twisting_stiffness = twisting_stiffness
        self.rods_info[rod_idx].plastic_yield = plastic_yield
        self.rods_info[rod_idx].plastic_creep = plastic_creep

        self.rods_info[rod_idx].is_loop = is_loop

        for i, i_b in ti.ndrange(self._n_rods, self._B):
            self.rods_energy[i, i_b].stretching_energy = 0.0
            self.rods_energy[i, i_b].bending_energy = 0.0
            self.rods_energy[i, i_b].twisting_energy = 0.0

        # -------------------------------- build indices --------------------------------

        self.rods_info[rod_idx].first_vert_idx = v_start
        self.rods_info[rod_idx].first_edge_idx = e_start
        self.rods_info[rod_idx].first_internal_vert_idx = iv_start
        self.rods_info[rod_idx].n_verts = n_verts

        # rod id of verts
        for i_v in range(n_verts):
            vert_idx = i_v + v_start
            self.vertices_info[vert_idx].rod_idx = rod_idx

        # vert id of edges
        n_edges = n_verts if is_loop else n_verts - 1
        for i_e in range(n_edges):
            vert_idx = i_e + v_start
            edge_idx = i_e + e_start
            self.edges_info[edge_idx].vert_idx = vert_idx

        # edge id of internal verts
        n_internal_verts = n_verts - (0 if is_loop else 2)
        for i_iv in range(n_internal_verts):
            edge_idx = -1
            if is_loop:
                edge_idx = tm.mod(i_iv - 1, n_internal_verts) + e_start
            else:
                edge_idx = i_iv + e_start
            iv_idx = i_iv + iv_start
            self.internal_vertices_info[iv_idx].edge_idx = edge_idx

    @ti.kernel
    def _kernel_finalize_rest_states(
        self,
        f: ti.i32,
        rod_idx: ti.i32,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        segment_mass: ti.f64,        # NOTE: we can use array
        segment_radius: ti.f64,      # NOTE: we can use array
        static_friction: ti.f64,     # NOTE: we can use array
        kinetic_friction: ti.f64,    # NOTE: we can use array
        restitution: ti.f64,         # NOTE: we can use array
        verts_rest: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        edges_rest: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    ):  
        n_verts_local = verts_rest.shape[0]
        for i_v in range(n_verts_local):
            i_global = i_v + v_start

            # info (static)
            self.vertices_info[i_global].mass = segment_mass
            self.vertices_info[i_global].radius = segment_radius
            self.vertices_info[i_global].mu_s = static_friction
            self.vertices_info[i_global].mu_k = kinetic_friction
            self.vertices_info[i_global].restitution = restitution
            self.vertices_info[i_global].rod_idx = rod_idx
            # finalize rest vertices    # not used
            # self.vertices_info[i_global].vert_rest = verts_rest[i_v]

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        ti.loop_config(serialize=True)
        for i_e in range(n_edges_local):
            i_global = i_e + e_start
            # v_s, v_e = self.get_edge_vertices(i_global)

            # finalize rest edges

            # self.edges_info[i_global].edge_rest = self.vertices_info[v_e].vert_rest - self.vertices_info[v_s].vert_rest
            self.edges_info[i_global].edge_rest = edges_rest[i_e]
            self.edges_info[i_global].length_rest = tm.length(self.edges_info[i_global].edge_rest)
            self.edges_info[i_global].d3_rest = self.edges_info[i_global].edge_rest.normalized()

            # finalize rest material frame (d1, d2, d3)

            if i_e == 0: # first edge
                self.edges_info[i_global].d1_rest = get_perpendicular_vector(self.edges_info[i_global].d3_rest)
            else:
                self.edges_info[i_global].d1_rest = parallel_transport_normalized(
                    self.edges_info[i_global - 1].d3_rest,
                    self.edges_info[i_global].d3_rest,
                    self.edges_info[i_global - 1].d1_rest,
                )
            self.edges_info[i_global].d2_rest = tm.cross(self.edges_info[i_global].d3_rest, self.edges_info[i_global].d1_rest)

        # deal with loop topology

        if self.rods_info[rod_idx].is_loop:
            e_end = e_start + n_edges_local - 1

            d1_final_transport = parallel_transport_normalized(
                self.edges_info[e_end].d3_rest,
                self.edges_info[e_start].d3_rest,
                self.edges_info[e_end].d1_rest,
            )

            total_holonomy_angle = get_angle(
                self.edges_info[e_start].d3_rest,
                d1_final_transport,
                self.edges_info[e_start].d1_rest,
            )

            for i_e in range(n_edges_local):
                i_global = i_e + e_start

                correction_angle = - total_holonomy_angle * (i_e / n_edges_local)
                d1_uncorrected = self.edges_info[i_global].d1_rest
                d2_uncorrected = self.edges_info[i_global].d2_rest
                c, s = ti.cos(correction_angle), ti.sin(correction_angle)
                self.edges_info[i_global].d1_rest = c * d1_uncorrected + s * d2_uncorrected
                self.edges_info[i_global].d2_rest = -s * d1_uncorrected + c * d2_uncorrected

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in ti.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # finalize rest curvature binormal

            rest_kbs = curvature_binormal(self.edges_info[e_s].d3_rest, self.edges_info[e_e].d3_rest)
            self.internal_vertices_ng[f, i_iv, i_b].kappa_rest = ti.Vector([
                  0.5 * tm.dot(rest_kbs, self.edges_info[e_s].d2_rest + self.edges_info[e_e].d2_rest),
                - 0.5 * tm.dot(rest_kbs, self.edges_info[e_s].d1_rest + self.edges_info[e_e].d1_rest),
            ])
            self.internal_vertices_info[i_global].twist_rest = 0.0  # assume no initial twist

    @ti.kernel
    def _kernel_finalize_states(
        self,
        f: ti.i32,
        rod_idx: ti.i32,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        fixed: ti.u1,
        verts: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        edges: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in ti.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start

            # state (dynamic)
            self.vertices[f, i_global, i_b].vert = verts[i_v]
            self.vertices[f, i_global, i_b].vel = ti.Vector.zero(gs.ti_float, 3)

            # state (dynamic w/o grad)
            self.vertices_ng[f, i_global, i_b].fixed = fixed
            self.vertices_ng[f, i_global, i_b].is_kinematic = False

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        for i_b in range(self._B):
            for i_e in range(n_edges_local):
                i_global = i_e + e_start
                # v_s, v_e = self.get_edge_vertices(i_global)

                # state (dynamic)

                # self.edges[f, i_global, i_b].edge = self.vertices[f, v_e, i_b].vert - self.vertices[f, v_s, i_b].vert
                self.edges_ng[f, i_global, i_b].edge = edges[i_e]
                self.edges_ng[f, i_global, i_b].length = tm.length(self.edges_ng[f, i_global, i_b].edge)
                self.edges_ng[f, i_global, i_b].d3 = self.edges_ng[f, i_global, i_b].edge.normalized()

                if i_e == 0: # first edge
                    self.edges_ng[f, i_global, i_b].d1 = get_perpendicular_vector(self.edges_ng[f, i_global, i_b].d3)
                else:
                    self.edges_ng[f, i_global, i_b].d1 = parallel_transport_normalized(
                        self.edges_ng[f, i_global - 1, i_b].d3,
                        self.edges_ng[f, i_global, i_b].d3,
                        self.edges_ng[f, i_global - 1, i_b].d1,
                    )
                self.edges_ng[f, i_global, i_b].d1_ref = self.edges_ng[f, i_global, i_b].d1

                self.edges_ng[f, i_global, i_b].d2 = tm.cross(self.edges_ng[f, i_global, i_b].d3, self.edges_ng[f, i_global, i_b].d1)
                self.edges_ng[f, i_global, i_b].d2_ref = self.edges_ng[f, i_global, i_b].d2

                self.edges[f, i_global, i_b].theta = 0.0  # assume no initial twist
                self.edges[f, i_global, i_b].omega = 0.0  # assume no initial twist rate

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in ti.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # state (dynamic)

            self.internal_vertices_ng[f, i_global, i_b].kb = curvature_binormal(
                self.edges_ng[f, e_s, i_b].d3, self.edges_ng[f, e_e, i_b].d3
            )
            self.internal_vertices_ng[f, i_global, i_b].twist = 0.0    # assume no initial twist

    @ti.kernel
    def _kernel_set_vertices_pos(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        pos: ti.types.ndarray(),
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices[f, i_global, i_b].vert[j] = pos[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_vertices_pos_grad(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        pos_grad: ti.types.ndarray(),
    ):  
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                pos_grad[i_b, i_v, j] = self.vertices.grad[f, i_global, i_b].vert[j]

    @ti.kernel
    def _kernel_set_vertices_vel(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices[f, i_global, i_b].vel[j] = vel[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_vertices_vel_grad(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        vel_grad: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                vel_grad[i_b, i_v, j] = self.vertices.grad[f, i_global, i_b].vel[j]

    @ti.kernel
    def _kernel_set_edges_theta(
        self,
        f: ti.i32,
        e_start: ti.i32,
        n_edges: ti.i32,
        theta: ti.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in ti.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            self.edges[f, i_global, i_b].theta = theta[i_b, i_e]

    @ti.kernel
    def _kernel_set_edges_theta_grad(
        self,
        f: ti.i32,
        e_start: ti.i32,
        n_edges: ti.i32,
        theta_grad: ti.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in ti.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            theta_grad[i_b, i_e] = self.edges.grad[f, i_global, i_b].theta

    @ti.kernel
    def _kernel_set_edges_omega(
        self,
        f: ti.i32,
        e_start: ti.i32,
        n_edges: ti.i32,
        omega: ti.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in ti.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            self.edges[f, i_global, i_b].omega = omega[i_b, i_e]

    @ti.kernel
    def _kernel_set_edges_omega_grad(
        self,
        f: ti.i32,
        e_start: ti.i32,
        n_edges: ti.i32,
        omega_grad: ti.types.ndarray(),  # shape [B, n_edges]
    ):
        for i_e, i_b in ti.ndrange(n_edges, self._B):
            i_global = i_e + e_start
            omega_grad[i_b, i_e] = self.edges.grad[f, i_global, i_b].omega

    @ti.kernel
    def _kernel_set_fixed_states(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        fixed: ti.types.ndarray(),  # shape [B, n_vertices]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            self.vertices_ng[f, i_global, i_b].fixed = fixed[i_b, i_v]

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),        # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),        # shape [B, n_vertices, 3]
        fixed: ti.types.ndarray(),      # shape [B, n_vertices]
        theta: ti.types.ndarray(),      # shape [B, n_edges]
        omega: ti.types.ndarray(),      # shape [B, n_edges]
        d1: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d2: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d3: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d1_ref: ti.types.ndarray(),     # shape [B, n_edges, 3]
        d2_ref: ti.types.ndarray(),     # shape [B, n_edges, 3]
        kb: ti.types.ndarray(),             # shape [B, n_internal_vertices, 3]
        twist: ti.types.ndarray(),          # shape [B, n_internal_vertices]
        kappa_rest: ti.types.ndarray(),     # shape [B, n_internal_vertices, 2]
    ):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self.vertices[f, i_v, i_b].vert[j]
                vel[i_b, i_v, j] = self.vertices[f, i_v, i_b].vel[j]
            fixed[i_b, i_v] = self.vertices_ng[f, i_v, i_b].fixed

        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            theta[i_b, i_e] = self.edges[f, i_e, i_b].theta
            omega[i_b, i_e] = self.edges[f, i_e, i_b].omega
            for j in ti.static(range(3)):
                d1[i_b, i_e, j] = self.edges_ng[f, i_e, i_b].d1[j]
                d2[i_b, i_e, j] = self.edges_ng[f, i_e, i_b].d2[j]
                d3[i_b, i_e, j] = self.edges_ng[f, i_e, i_b].d3[j]
                d1_ref[i_b, i_e, j] = self.edges_ng[f, i_e, i_b].d1_ref[j]
                d2_ref[i_b, i_e, j] = self.edges_ng[f, i_e, i_b].d2_ref[j]

        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            for j in ti.static(range(3)):
                kb[i_b, i_iv, j] = self.internal_vertices_ng[f, i_iv, i_b].kb[j]
            twist[i_b, i_iv] = self.internal_vertices_ng[f, i_iv, i_b].twist
            for j in ti.static(range(2)):
                kappa_rest[i_b, i_iv, j] = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[j]

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                pos_j = ti.cast(self.vertices[f, i_v, i_b].vert[j], ti.f32)
                self.vertices_render[i_v, i_b][j] = pos_j + self.envs_offset[i_b][j]

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),        # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),        # shape [B, n_vertices, 3]
        fixed: ti.types.ndarray(),      # shape [B, n_vertices]
        theta: ti.types.ndarray(),      # shape [B, n_edges]
        omega: ti.types.ndarray(),      # shape [B, n_edges]
        d1: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d2: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d3: ti.types.ndarray(),         # shape [B, n_edges, 3]
        d1_ref: ti.types.ndarray(),     # shape [B, n_edges, 3]
        d2_ref: ti.types.ndarray(),     # shape [B, n_edges, 3]
        kb: ti.types.ndarray(),             # shape [B, n_internal_vertices, 3]
        twist: ti.types.ndarray(),          # shape [B, n_internal_vertices]
        kappa_rest: ti.types.ndarray(),     # shape [B, n_internal_vertices, 2]
    ):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for j in ti.static(range(3)):
                self.vertices[f, i_v, i_b].vert[j] = pos[i_b, i_v, j]
                self.vertices[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]
            self.vertices_ng[f, i_v, i_b].fixed = fixed[i_b, i_v]

        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges[f, i_e, i_b].theta = theta[i_b, i_e]
            self.edges[f, i_e, i_b].omega = omega[i_b, i_e]

            for j in ti.static(range(3)):
                self.edges_ng[f, i_e, i_b].d1[j] = d1[i_b, i_e, j]
                self.edges_ng[f, i_e, i_b].d2[j] = d2[i_b, i_e, j]
                self.edges_ng[f, i_e, i_b].d3[j] = d3[i_b, i_e, j]
                self.edges_ng[f, i_e, i_b].d1_ref[j] = d1_ref[i_b, i_e, j]
                self.edges_ng[f, i_e, i_b].d2_ref[j] = d2_ref[i_b, i_e, j]

            # NOTE: we compute intermediate variables so do not need to store them
            v_s, v_e = self.get_edge_vertices(i_e)
            self.edges_ng[f, i_e, i_b].edge = self.vertices[f, v_e, i_b].vert - self.vertices[f, v_s, i_b].vert
            self.edges_ng[f, i_e, i_b].length = tm.length(self.edges_ng[f, i_e, i_b].edge)

        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            for j in ti.static(range(3)):
                self.internal_vertices_ng[f, i_iv, i_b].kb[j] = kb[i_b, i_iv, j]
            self.internal_vertices_ng[f, i_iv, i_b].twist = twist[i_b, i_iv]
            for j in ti.static(range(2)):
                self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[j] = kappa_rest[i_b, i_iv, j]

    # ------------------------------------------------------------------------------------
    # --------------------------------- index utilities -----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def get_edge_vertices(self, i_e: ti.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_end = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            next_local_idx = ti.cast(tm.mod(local_v_start + 1, n_verts), ti.i32)
            v_end = first_vert_idx + next_local_idx
        else:
            v_end = v_start + 1

        return v_start, v_end

    @ti.func
    def get_hinge_edges(self, i_iv: ti.i32):
        e_start = self.internal_vertices_info[i_iv].edge_idx
        v_start_of_e_start = self.edges_info[e_start].vert_idx
        rod_id = self.vertices_info[v_start_of_e_start].rod_idx

        e_end = -1
        if self.rods_info[rod_id].is_loop:
            first_edge_idx = self.rods_info[rod_id].first_edge_idx
            n_verts = self.rods_info[rod_id].n_verts
            n_edges = n_verts - 1   # normal case
            if self.rods_info[rod_id].is_loop:
                n_edges = n_verts

            local_e_start = e_start - first_edge_idx
            next_local_idx = ti.cast(tm.mod(local_e_start + 1, n_edges), ti.i32)
            e_end = first_edge_idx + next_local_idx
        else:
            e_end = e_start + 1

        return e_start, e_end

    @ti.func
    def get_hinge_vertices(self, i_e: ti.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_middle, v_end = -1, -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            local_v_middle = ti.cast(tm.mod(local_v_start + 1, n_verts), ti.i32)
            local_v_end = ti.cast(tm.mod(local_v_start + 2, n_verts), ti.i32)

            v_middle = first_vert_idx + local_v_middle
            v_end = first_vert_idx + local_v_end
        else:
            v_middle = v_start + 1
            v_end = v_start + 2

        return v_start, v_middle, v_end

    @ti.func
    def get_next_vertex_of_edge(self, i_v: ti.i32):
        rod_id = self.vertices_info[i_v].rod_idx

        ip1_v = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_i_v = i_v - first_vert_idx
            next_local_idx = ti.cast(tm.mod(local_i_v + 1, n_verts), ti.i32)
            ip1_v = first_vert_idx + next_local_idx
        else:
            ip1_v = i_v + 1

        return ip1_v

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height
    
    @property
    def damping(self):
        return self._damping

    @property
    def angular_damping(self):
        return self._angular_damping

    @property
    def n_dofs(self):
        return sum([entity.n_dofs for entity in self._entities])
    
    @property
    def n_rods(self):
        return len(self._entities)

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_edges(self):
        return sum([entity.n_edges for entity in self._entities])

    @property
    def n_internal_vertices(self):
        return sum([entity.n_internal_vertices for entity in self._entities])

    # ------------------------------------------------------------------------------------
    # -------------------------------- pbd constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def _func_get_inverse_mass(self, f: ti.i32, i_v: ti.i32, i_b: ti.i32):
        mass = self.vertices_info[i_v].mass
        inv_mass = 0.0
        if (
            self.vertices_ng[f, i_v, i_b].fixed or 
            self.vertices_ng[f, i_v, i_b].is_kinematic or 
            mass <= 0.
        ):
            inv_mass = 0.0
        else:
            inv_mass = 1.0 / mass
        return inv_mass

    @ti.kernel
    def _kernel_clear_contact_states(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            for j in ti.static(range(3)):
                self.rr_constraints[f, i_p, i_b].normal[j] = 0.0
            self.rr_constraints[f, i_p, i_b].penetration = 0.0

    @ti.kernel
    def _kernel_clear_contact_states_all_substeps(self):
        for i_f, i_p, i_b in ti.ndrange(self._sim.substeps_local, self._n_valid_edge_pairs, self._B):
            for j in ti.static(range(3)):
                self.rr_constraints[i_f, i_p, i_b].normal[j] = 0.0
            self.rr_constraints[i_f, i_p, i_b].penetration = 0.0

    @ti.kernel
    def _kernel_clear_kinematic_states(self, f: ti.i32):
        # TODO: do we need to clear kinematic states?
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices_ng[f, i_v, i_b].is_kinematic = False

    @ti.kernel
    def _kernel_clear_kinematic_states_all_substeps(self):
        for i_f, i_v, i_b in ti.ndrange(self._sim.substeps_local, self._n_vertices, self._B):
            self.vertices_ng[i_f, i_v, i_b].is_kinematic = False

    @ti.kernel
    def _kernel_apply_inextensibility_constraints(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check inextensibility enabled
            if self.rods_info[rod_id].use_inextensible:
                inv_mass_s = self._func_get_inverse_mass(f, v_s, i_b)
                inv_mass_e = self._func_get_inverse_mass(f, v_e, i_b)
                inv_mass_sum = inv_mass_s + inv_mass_e

                if inv_mass_sum > EPS:
                    p_s, p_e = self.vertices[f + 1, v_s, i_b].vert, self.vertices[f + 1, v_e, i_b].vert

                    edge_vec = p_e - p_s
                    dist = tm.length(edge_vec)

                    constraint_error = dist - self.edges_info[i_e].length_rest

                    if dist > EPS:
                        normal = edge_vec / dist
                        lambda_ = constraint_error / inv_mass_sum
                        delta_p_s = lambda_ * inv_mass_s * normal
                        delta_p_e = -lambda_ * inv_mass_e * normal

                        # apply corrections
                        self.vertices[f + 1, v_s, i_b].vert += delta_p_s
                        self.vertices[f + 1, v_e, i_b].vert += delta_p_e

    @ti.kernel
    def _kernel_apply_rod_collision_constraints(self, f: ti.i32, iter_idx: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
            idx_a2 = self.get_next_vertex_of_edge(idx_a1)
            idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
            idx_b2 = self.get_next_vertex_of_edge(idx_b1)

            p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
            p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

            radius_a = (self.vertices_info[idx_a1].radius + self.vertices_info[idx_a2].radius) * 0.5
            radius_b = (self.vertices_info[idx_b1].radius + self.vertices_info[idx_b2].radius) * 0.5

            # compute closest points (t, u) and distance
            e1, e2 = p_a2 - p_a1, p_b2 - p_b1
            e12 = p_b1 - p_a1
            d1, d2 = e1.dot(e1), e2.dot(e2)
            r = e1.dot(e2)
            s1, s2 = e1.dot(e12), e2.dot(e12)
            den = d1 * d2 - r * r

            t = 0.0
            if den > EPS:
                t = (s1 * d2 - s2 * r) / den
            t = tm.clamp(t, 0.0, 1.0)

            u_unclamped = 0.0
            if d2 > EPS:
                u_unclamped = (t * r - s2) / d2
            u = tm.clamp(u_unclamped, 0.0, 1.0)

            # re-compute t if u was clamped
            if ti.abs(u - u_unclamped) > EPS:
                if d1 > EPS:
                    t = (u * r + s1) / d1
                t = tm.clamp(t, 0.0, 1.0)

            # check for penetration
            closest_p_a = p_a1 + t * e1
            closest_p_b = p_b1 + u * e2
            dist_vec = closest_p_a - closest_p_b
            dist = tm.length(dist_vec)

            penetration = radius_a + radius_b - dist
            if penetration > 0.:
                normal = dist_vec.normalized() if dist > EPS else ti.Vector([0.0, 0.0, 1.0])

                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = tm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    lambda_ = penetration / w_sum_sq_inv_mass

                    self.vertices[f + 1, idx_a1, i_b].vert += lambda_ * im[0] * w[0] * normal
                    self.vertices[f + 1, idx_a2, i_b].vert += lambda_ * im[1] * w[1] * normal
                    self.vertices[f + 1, idx_b1, i_b].vert -= lambda_ * im[2] * w[2] * normal
                    self.vertices[f + 1, idx_b2, i_b].vert -= lambda_ * im[3] * w[3] * normal

                if iter_idx == 0:
                    self.rr_constraints[f, i_p, i_b].normal = normal
                    self.rr_constraints[f, i_p, i_b].penetration = penetration

    @ti.kernel
    def _kernel_apply_rod_collision_constraints_grad(self, f: ti.i32, iter_idx: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
            idx_a2 = self.get_next_vertex_of_edge(idx_a1)
            idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
            idx_b2 = self.get_next_vertex_of_edge(idx_b1)

            p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
            p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

            radius_a = (self.vertices_info[idx_a1].radius + self.vertices_info[idx_a2].radius) * 0.5
            radius_b = (self.vertices_info[idx_b1].radius + self.vertices_info[idx_b2].radius) * 0.5

            # compute closest points (t, u) and distance
            e1, e2 = p_a2 - p_a1, p_b2 - p_b1
            e12 = p_b1 - p_a1
            d1, d2 = e1.dot(e1), e2.dot(e2)
            r = e1.dot(e2)
            s1, s2 = e1.dot(e12), e2.dot(e12)
            den = d1 * d2 - r * r

            t = 0.0
            if den > EPS:
                t = (s1 * d2 - s2 * r) / den
            t = tm.clamp(t, 0.0, 1.0)

            u_unclamped = 0.0
            if d2 > EPS:
                u_unclamped = (t * r - s2) / d2
            u = tm.clamp(u_unclamped, 0.0, 1.0)

            # re-compute t if u was clamped
            if ti.abs(u - u_unclamped) > EPS:
                if d1 > EPS:
                    t = (u * r + s1) / d1
                t = tm.clamp(t, 0.0, 1.0)

            # check for penetration
            closest_p_a = p_a1 + t * e1
            closest_p_b = p_b1 + u * e2
            dist_vec = closest_p_a - closest_p_b
            dist = tm.length(dist_vec)

            penetration = radius_a + radius_b - dist
            if penetration > 0.:
                g_p_a1 = self.vertices.grad[f + 1, idx_a1, i_b].vert
                g_p_a2 = self.vertices.grad[f + 1, idx_a2, i_b].vert
                g_p_b1 = self.vertices.grad[f + 1, idx_b1, i_b].vert
                g_p_b2 = self.vertices.grad[f + 1, idx_b2, i_b].vert

                normal = dist_vec.normalized() if dist > EPS else ti.Vector([0.0, 0.0, 1.0])
                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = tm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    g_displacement_vec = (
                        im[0] * w[0] * g_p_a1 + im[1] * w[1] * g_p_a2 -
                        im[2] * w[2] * g_p_b1 - im[3] * w[3] * g_p_b2
                    )

                    g_lambda = normal.dot(g_displacement_vec)
                    g_penetration = g_lambda / w_sum_sq_inv_mass

                    if iter_idx == 0:
                        g_penetration += self.rr_constraints.grad[f, i_p, i_b].penetration

                    g_dist = -g_penetration
                    g_dist_vec = g_dist * normal

                    g_closest_p_a = g_dist_vec
                    g_closest_p_b = -g_dist_vec

                    # ratio-preserve distribution
                    g_v_a1 = (1.0 - t) * g_closest_p_a
                    g_v_a2 = t * g_closest_p_a
                    g_v_b1 = (1.0 - u) * g_closest_p_b
                    g_v_b2 = u * g_closest_p_b

                    total_mag_sq = (
                        g_v_a1.dot(g_v_a1) + g_v_a2.dot(g_v_a2) +
                        g_v_b1.dot(g_v_b1) + g_v_b2.dot(g_v_b2)
                    )
                    scale = 1.0
                    if total_mag_sq > self._max_collision_grad_norm ** 2:
                        total_mag = tm.sqrt(total_mag_sq)
                        if total_mag > EPS:
                            scale = self._max_collision_grad_norm / total_mag

                    self.vertices.grad[f + 1, idx_a1, i_b].vert += g_v_a1 * scale
                    self.vertices.grad[f + 1, idx_a2, i_b].vert += g_v_a2 * scale
                    self.vertices.grad[f + 1, idx_b1, i_b].vert += g_v_b1 * scale
                    self.vertices.grad[f + 1, idx_b2, i_b].vert += g_v_b2 * scale

    @ti.kernel
    def _kernel_apply_rod_friction(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            penetration = self.rr_constraints[f, i_p, i_b].penetration
            if penetration > 0.0:
                idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
                idx_a2 = self.get_next_vertex_of_edge(idx_a1)
                idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
                idx_b2 = self.get_next_vertex_of_edge(idx_b1)

                p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
                p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

                # compute closest points (t, u) and distance
                e1, e2 = p_a2 - p_a1, p_b2 - p_b1
                e12 = p_b1 - p_a1
                d1, d2 = e1.dot(e1), e2.dot(e2)
                r = e1.dot(e2)
                s1, s2 = e1.dot(e12), e2.dot(e12)
                den = d1 * d2 - r * r

                t = 0.0
                if den > EPS:
                    t = (s1 * d2 - s2 * r) / den
                t = tm.clamp(t, 0.0, 1.0)

                u_unclamped = 0.0
                if d2 > EPS:
                    u_unclamped = (t * r - s2) / d2
                u = tm.clamp(u_unclamped, 0.0, 1.0)

                # Re-compute t if u was clamped
                if ti.abs(u - u_unclamped) > EPS:
                    if d1 > EPS:
                        t = (u * r + s1) / d1
                    t = tm.clamp(t, 0.0, 1.0)

                v_a1, v_a2 = self.vertices[f + 1, idx_a1, i_b].vel, self.vertices[f + 1, idx_a2, i_b].vel
                v_b1, v_b2 = self.vertices[f + 1, idx_b1, i_b].vel, self.vertices[f + 1, idx_b2, i_b].vel

                v_a = (1 - t) * v_a1 + t * v_a2
                v_b = (1 - u) * v_b1 + u * v_b2
                v_rel = v_a - v_b

                normal = self.rr_constraints[f, i_p, i_b].normal
                v_normal_mag = v_rel.dot(normal)
                v_tangent = v_rel - v_normal_mag * normal
                v_tangent_norm = tm.length(v_tangent)

                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = tm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    normal_vel_mag = penetration / self._substep_dt

                    mu_s = (self.vertices_info[idx_a1].mu_s + self.vertices_info[idx_a2].mu_s + self.vertices_info[idx_b1].mu_s + self.vertices_info[idx_b2].mu_s) * 0.25
                    mu_k = (self.vertices_info[idx_a1].mu_k + self.vertices_info[idx_a2].mu_k + self.vertices_info[idx_b1].mu_k + self.vertices_info[idx_b2].mu_k) * 0.25

                    delta_v_tangent = ti.Vector.zero(gs.ti_float, 3)
                    if v_tangent_norm < mu_s * normal_vel_mag:
                        delta_v_tangent = -v_tangent
                    else:
                        delta_v_tangent = -v_tangent.normalized() * mu_k * normal_vel_mag

                    lambda_ = delta_v_tangent / w_sum_sq_inv_mass
                    self.vertices[f + 1, idx_a1, i_b].vel += lambda_ * im[0] * w[0]
                    self.vertices[f + 1, idx_a2, i_b].vel += lambda_ * im[1] * w[1]
                    self.vertices[f + 1, idx_b1, i_b].vel -= lambda_ * im[2] * w[2]
                    self.vertices[f + 1, idx_b2, i_b].vel -= lambda_ * im[3] * w[3]

    @ti.kernel
    def _kernel_apply_rod_friction_grad(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            if self.rr_constraints[f, i_p, i_b].penetration > 0.0:
                idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
                idx_a2 = self.get_next_vertex_of_edge(idx_a1)
                idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
                idx_b2 = self.get_next_vertex_of_edge(idx_b1)

                p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
                p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

                # compute closest points (t, u) and distance
                e1, e2 = p_a2 - p_a1, p_b2 - p_b1
                e12 = p_b1 - p_a1
                d1, d2 = e1.dot(e1), e2.dot(e2)
                r = e1.dot(e2)
                s1, s2 = e1.dot(e12), e2.dot(e12)
                den = d1 * d2 - r * r

                t = 0.0
                if den > EPS:
                    t = (s1 * d2 - s2 * r) / den
                t = tm.clamp(t, 0.0, 1.0)

                u_unclamped = 0.0
                if d2 > EPS:
                    u_unclamped = (t * r - s2) / d2
                u = tm.clamp(u_unclamped, 0.0, 1.0)

                # Re-compute t if u was clamped
                if ti.abs(u - u_unclamped) > EPS:
                    if d1 > EPS:
                        t = (u * r + s1) / d1
                    t = tm.clamp(t, 0.0, 1.0)

                g_v_a1_out = self.vertices.grad[f + 1, idx_a1, i_b].vel
                g_v_a2_out = self.vertices.grad[f + 1, idx_a2, i_b].vel
                g_v_b1_out = self.vertices.grad[f + 1, idx_b1, i_b].vel
                g_v_b2_out = self.vertices.grad[f + 1, idx_b2, i_b].vel

                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b), self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b), self._func_get_inverse_mass(f, idx_b2, i_b),
                ])
                w_sum_sq_inv_mass = tm.dot(w * w, im)

                if w_sum_sq_inv_mass > EPS:
                    g_lambda_vec = (
                        g_v_a1_out * im[0] * w[0] + g_v_a2_out * im[1] * w[1] -
                        g_v_b1_out * im[2] * w[2] - g_v_b2_out * im[3] * w[3]
                    )

                    g_delta_v_tangent = g_lambda_vec / w_sum_sq_inv_mass

                    v_a1, v_a2 = self.vertices[f + 1, idx_a1, i_b].vel, self.vertices[f + 1, idx_a2, i_b].vel
                    v_b1, v_b2 = self.vertices[f + 1, idx_b1, i_b].vel, self.vertices[f + 1, idx_b2, i_b].vel
                    v_rel = ((1 - t) * v_a1 + t * v_a2) - ((1 - u) * v_b1 + u * v_b2)
                    normal = self.rr_constraints[f, i_p, i_b].normal
                    v_tangent = v_rel - v_rel.dot(normal) * normal
                    v_tangent_norm = tm.length(v_tangent)

                    penetration = self.rr_constraints[f, i_p, i_b].penetration
                    normal_vel_mag = penetration / self._substep_dt
                    mu_s = (self.vertices_info[idx_a1].mu_s + self.vertices_info[idx_a2].mu_s + self.vertices_info[idx_b1].mu_s + self.vertices_info[idx_b2].mu_s) * 0.25
                    mu_k = (self.vertices_info[idx_a1].mu_k + self.vertices_info[idx_a2].mu_k + self.vertices_info[idx_b1].mu_k + self.vertices_info[idx_b2].mu_k) * 0.25

                    g_v_tangent = ti.Vector.zero(gs.ti_float, 3)
                    g_normal_vel_mag = 0.0

                    if v_tangent_norm < mu_s * normal_vel_mag:
                        g_v_tangent -= g_delta_v_tangent
                    else: # Differentiate through the kinetic friction case
                        n_t = v_tangent.normalized()
                        F_k = mu_k * normal_vel_mag
                        g_F_k = -n_t.dot(g_delta_v_tangent)
                        g_n_t = -F_k * g_delta_v_tangent
                        inv_norm = 1.0 / tm.max(v_tangent_norm, EPS)
                        g_v_tangent += (g_n_t - n_t.dot(g_n_t) * n_t) * inv_norm
                        g_normal_vel_mag += g_F_k * mu_k
                    
                    self.rr_constraints.grad[f, i_p, i_b].penetration += g_normal_vel_mag / self._substep_dt
                    
                    g_v_rel = g_v_tangent - normal.dot(g_v_tangent) * normal
                    
                    g_v_a = g_v_rel
                    g_v_b = -g_v_rel

                    # ratio-preserve distribution
                    g_v_a1 = (1.0 - t) * g_v_a
                    g_v_a2 = t * g_v_a
                    g_v_b1 = (1.0 - u) * g_v_b
                    g_v_b2 = u * g_v_b

                    total_mag_sq = (
                        g_v_a1.dot(g_v_a1) + g_v_a2.dot(g_v_a2) +
                        g_v_b1.dot(g_v_b1) + g_v_b2.dot(g_v_b2)
                    )
                    scale = 1.0
                    if total_mag_sq > self._max_collision_grad_norm ** 2:
                        total_mag = tm.sqrt(total_mag_sq)
                        if total_mag > EPS:
                            scale = self._max_collision_grad_norm / total_mag

                    self.vertices.grad[f + 1, idx_a1, i_b].vel += g_v_a1 * scale
                    self.vertices.grad[f + 1, idx_a2, i_b].vel += g_v_a2 * scale
                    self.vertices.grad[f + 1, idx_b1, i_b].vel += g_v_b1 * scale
                    self.vertices.grad[f + 1, idx_b2, i_b].vel += g_v_b2 * scale

    @ti.ad.grad_replaced
    def collision_forward(self, f, iter_idx):
        self._kernel_apply_rod_collision_constraints(f, iter_idx)

    @ti.ad.grad_for(collision_forward)
    def collision_backward(self, f, iter_idx):
        # self._kernel_apply_rod_collision_constraints_grad(f, iter_idx)
        pass # NOTE: just ignore also works

    @ti.ad.grad_replaced
    def friction_forward(self, f):
        self._kernel_apply_rod_friction(f)

    @ti.ad.grad_for(friction_forward)
    def friction_backward(self, f):
        # self._kernel_apply_rod_friction_grad(f)
        pass # NOTE: just ignore also works

    # ------------------------- NOTE: Smooth version (may be less stable) -------------------------

    # @ti.kernel
    # def _kernel_apply_inextensibility_constraints(self, f: ti.i32):
    #     for i_e, i_b in ti.ndrange(self._n_edges, self._B):
    #         v_s, v_e = self.get_edge_vertices(i_e)
    #         rod_id = self.vertices_info[v_s].rod_idx

    #         # check inextensibility enabled
    #         if self.rods_info[rod_id].use_inextensible:
    #             inv_mass_s = self._func_get_inverse_mass(f, v_s, i_b)
    #             inv_mass_e = self._func_get_inverse_mass(f, v_e, i_b)
    #             inv_mass_sum = inv_mass_s + inv_mass_e

    #             p_s, p_e = self.vertices[f + 1, v_s, i_b].vert, self.vertices[f + 1, v_e, i_b].vert

    #             edge_vec = p_e - p_s
    #             dist = tm.length(edge_vec)

    #             constraint_error = dist - self.edges_info[i_e].length_rest

    #             normal = edge_vec / (dist + EPS)
    #             lambda_ = constraint_error / (inv_mass_sum + EPS)
    #             delta_p_s = lambda_ * inv_mass_s * normal
    #             delta_p_e = -lambda_ * inv_mass_e * normal

    #             # apply corrections
    #             self.vertices[f + 1, v_s, i_b].vert += delta_p_s
    #             self.vertices[f + 1, v_e, i_b].vert += delta_p_e

    # @ti.kernel
    # def _kernel_apply_rod_collision_constraints(self, f: ti.i32, iter_idx: ti.i32):
    #     for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
    #         idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
    #         idx_a2 = self.get_next_vertex_of_edge(idx_a1)
    #         idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
    #         idx_b2 = self.get_next_vertex_of_edge(idx_b1)

    #         p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
    #         p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

    #         radius_a = (self.vertices_info[idx_a1].radius + self.vertices_info[idx_a2].radius) * 0.5
    #         radius_b = (self.vertices_info[idx_b1].radius + self.vertices_info[idx_b2].radius) * 0.5

    #         # compute closest points (t, u) and distance
    #         e1, e2 = p_a2 - p_a1, p_b2 - p_b1
    #         e12 = p_b1 - p_a1
    #         d1, d2 = e1.dot(e1), e2.dot(e2)
    #         r = e1.dot(e2)
    #         s1, s2 = e1.dot(e12), e2.dot(e12)
    #         den = d1 * d2 - r * r

    #         # t = 0.0
    #         # if den > EPS:
    #         #     t = (s1 * d2 - s2 * r) / den
    #         # t = tm.clamp(t, 0.0, 1.0)

    #         # u_unclamped = 0.0
    #         # if d2 > EPS:
    #         #     u_unclamped = (t * r - s2) / d2
    #         # u = tm.clamp(u_unclamped, 0.0, 1.0)

    #         # # re-compute t if u was clamped
    #         # if ti.abs(u - u_unclamped) > EPS:
    #         #     if d1 > EPS:
    #         #         t = (u * r + s1) / d1
    #         #     t = tm.clamp(t, 0.0, 1.0)

    #         t = tm.clamp((s1 * d2 - s2 * r) / (den + EPS), 0.0, 1.0)
    #         u = tm.clamp((t * r - s2) / (d2 + EPS), 0.0, 1.0)
    #         t = tm.clamp((u * r + s1) / (d1 + EPS), 0.0, 1.0) # Re-clamp t

    #         # check for penetration
    #         closest_p_a = p_a1 + t * e1
    #         closest_p_b = p_b1 + u * e2
    #         dist_vec = closest_p_a - closest_p_b
    #         dist = tm.length(dist_vec)

    #         penetration = radius_a + radius_b - dist
    #         clamped_penetration = tm.max(0., penetration)
    #         normal = dist_vec / (dist + EPS)

    #         w = ti.Vector([1.0 - t, t, 1.0 - u, u])
    #         im = ti.Vector([
    #             self._func_get_inverse_mass(f, idx_a1, i_b),
    #             self._func_get_inverse_mass(f, idx_a2, i_b),
    #             self._func_get_inverse_mass(f, idx_b1, i_b),
    #             self._func_get_inverse_mass(f, idx_b2, i_b),
    #         ])

    #         w_sum_sq_inv_mass = tm.dot(w * w, im)
    #         lambda_ = clamped_penetration / (w_sum_sq_inv_mass + EPS)

    #         self.vertices[f + 1, idx_a1, i_b].vert += lambda_ * im[0] * w[0] * normal
    #         self.vertices[f + 1, idx_a2, i_b].vert += lambda_ * im[1] * w[1] * normal
    #         self.vertices[f + 1, idx_b1, i_b].vert -= lambda_ * im[2] * w[2] * normal
    #         self.vertices[f + 1, idx_b2, i_b].vert -= lambda_ * im[3] * w[3] * normal

    #         if iter_idx == 0:
    #             self.rr_constraints[f, i_p, i_b].normal = normal
    #             self.rr_constraints[f, i_p, i_b].penetration = penetration

    # @ti.kernel
    # def _kernel_apply_rod_friction(self, f: ti.i32):
    #     for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
    #         penetration = self.rr_constraints[f, i_p, i_b].penetration
    #         clamped_penetration = ti.max(0, penetration)

    #         if clamped_penetration > 0.:
    #             idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
    #             idx_a2 = self.get_next_vertex_of_edge(idx_a1)
    #             idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
    #             idx_b2 = self.get_next_vertex_of_edge(idx_b1)

    #             p_a1, p_a2 = self.vertices[f + 1, idx_a1, i_b].vert, self.vertices[f + 1, idx_a2, i_b].vert
    #             p_b1, p_b2 = self.vertices[f + 1, idx_b1, i_b].vert, self.vertices[f + 1, idx_b2, i_b].vert

    #             # compute closest points (t, u) and distance
    #             e1, e2 = p_a2 - p_a1, p_b2 - p_b1
    #             e12 = p_b1 - p_a1
    #             d1, d2 = e1.dot(e1), e2.dot(e2)
    #             r = e1.dot(e2)
    #             s1, s2 = e1.dot(e12), e2.dot(e12)
    #             den = d1 * d2 - r * r

    #             # t = 0.0
    #             # if den > EPS:
    #             #     t = (s1 * d2 - s2 * r) / den
    #             # t = tm.clamp(t, 0.0, 1.0)

    #             # u_unclamped = 0.0
    #             # if d2 > EPS:
    #             #     u_unclamped = (t * r - s2) / d2
    #             # u = tm.clamp(u_unclamped, 0.0, 1.0)

    #             # # Re-compute t if u was clamped
    #             # if ti.abs(u - u_unclamped) > EPS:
    #             #     if d1 > EPS:
    #             #         t = (u * r + s1) / d1
    #             #     t = tm.clamp(t, 0.0, 1.0)

    #             t = tm.clamp((s1 * d2 - s2 * r) / (den + EPS), 0.0, 1.0)
    #             u = tm.clamp((t * r - s2) / (d2 + EPS), 0.0, 1.0)
    #             t = tm.clamp((u * r + s1) / (d1 + EPS), 0.0, 1.0) # Re-clamp t

    #             v_a1, v_a2 = self.vertices[f + 1, idx_a1, i_b].vel, self.vertices[f + 1, idx_a2, i_b].vel
    #             v_b1, v_b2 = self.vertices[f + 1, idx_b1, i_b].vel, self.vertices[f + 1, idx_b2, i_b].vel

    #             v_a = (1 - t) * v_a1 + t * v_a2
    #             v_b = (1 - u) * v_b1 + u * v_b2
    #             v_rel = v_a - v_b

    #             normal = self.rr_constraints[f, i_p, i_b].normal
    #             v_normal_mag = v_rel.dot(normal)
    #             v_tangent = v_rel - v_normal_mag * normal
    #             v_tangent_norm = tm.length(v_tangent)

    #             w = ti.Vector([1.0 - t, t, 1.0 - u, u])
    #             im = ti.Vector([
    #                 self._func_get_inverse_mass(f, idx_a1, i_b),
    #                 self._func_get_inverse_mass(f, idx_a2, i_b),
    #                 self._func_get_inverse_mass(f, idx_b1, i_b),
    #                 self._func_get_inverse_mass(f, idx_b2, i_b),
    #             ])

    #             w_sum_sq_inv_mass = tm.dot(w * w, im)
    #             normal_vel_mag = clamped_penetration / self._substep_dt

    #             mu_s = (self.vertices_info[idx_a1].mu_s + self.vertices_info[idx_a2].mu_s + self.vertices_info[idx_b1].mu_s + self.vertices_info[idx_b2].mu_s) * 0.25
    #             mu_k = (self.vertices_info[idx_a1].mu_k + self.vertices_info[idx_a2].mu_k + self.vertices_info[idx_b1].mu_k + self.vertices_info[idx_b2].mu_k) * 0.25

    #             delta_v_tangent = ti.Vector.zero(gs.ti_float, 3)
    #             if v_tangent_norm < mu_s * normal_vel_mag:
    #                 delta_v_tangent = -v_tangent
    #             else:
    #                 delta_v_tangent = -v_tangent.normalized() * mu_k * normal_vel_mag

    #             lambda_ = delta_v_tangent / (w_sum_sq_inv_mass + EPS)
    #             self.vertices[f + 1, idx_a1, i_b].vel += lambda_ * im[0] * w[0]
    #             self.vertices[f + 1, idx_a2, i_b].vel += lambda_ * im[1] * w[1]
    #             self.vertices[f + 1, idx_b1, i_b].vel -= lambda_ * im[2] * w[2]
    #             self.vertices[f + 1, idx_b2, i_b].vel -= lambda_ * im[3] * w[3]

    # ------------------------- NOTE: Smooth version (may be less stable) -------------------------

    # @ti.kernel
    # def _kernel_self_collision(self, f: ti.i32):    # not used
    #     """
    #     Use impulse based method to resolve self-collision.
    #     """

    #     for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
    #         idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
    #         idx_a2 = self.get_next_vertex_of_edge(idx_a1)
    #         idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
    #         idx_b2 = self.get_next_vertex_of_edge(idx_b1)

    #         p_a1, p_a2 = self.vertices[f, idx_a1, i_b].vert, self.vertices[f, idx_a2, i_b].vert
    #         p_b1, p_b2 = self.vertices[f, idx_b1, i_b].vert, self.vertices[f, idx_b2, i_b].vert

    #         radius_a = (self.vertices_info[idx_a1].radius + self.vertices_info[idx_a2].radius) * 0.5
    #         radius_b = (self.vertices_info[idx_b1].radius + self.vertices_info[idx_b2].radius) * 0.5

    #         # compute closest points (t, u) and distance
    #         e1, e2 = p_a2 - p_a1, p_b2 - p_b1
    #         e12 = p_b1 - p_a1
    #         d1, d2 = e1.dot(e1), e2.dot(e2)
    #         r = e1.dot(e2)
    #         s1, s2 = e1.dot(e12), e2.dot(e12)
    #         den = d1 * d2 - r * r

    #         t = 0.0
    #         if den > EPS:
    #             t = (s1 * d2 - s2 * r) / den
    #         t = tm.clamp(t, 0.0, 1.0)

    #         u_unclamped = 0.0
    #         if d2 > EPS:
    #             u_unclamped = (t * r - s2) / d2
    #         u = tm.clamp(u_unclamped, 0.0, 1.0)

    #         # re-compute t if u was clamped
    #         if ti.abs(u - u_unclamped) > EPS:
    #             if d1 > EPS:
    #                 t = (u * r + s1) / d1
    #             t = tm.clamp(t, 0.0, 1.0)

    #         # check for penetration
    #         closest_p_a = p_a1 + t * e1
    #         closest_p_b = p_b1 + u * e2
    #         dist_vec = closest_p_a - closest_p_b
    #         dist = dist_vec.norm(gs.EPS)
    #         penetration = radius_a + radius_b - dist

    #         if penetration > 0.:
    #             normal = dist_vec.normalized() if dist > EPS else ti.Vector([0.0, 0.0, 1.0])

    #             v_a1 = self.vertices[f, idx_a1, i_b].vel
    #             v_a2 = self.vertices[f, idx_a2, i_b].vel
    #             v_b1 = self.vertices[f, idx_b1, i_b].vel
    #             v_b2 = self.vertices[f, idx_b2, i_b].vel

    #             inv_mass_a1 = self._func_get_inverse_mass(f, idx_a1, i_b)
    #             inv_mass_a2 = self._func_get_inverse_mass(f, idx_a2, i_b)
    #             inv_mass_b1 = self._func_get_inverse_mass(f, idx_b1, i_b)
    #             inv_mass_b2 = self._func_get_inverse_mass(f, idx_b2, i_b)

    #             v_a = (1 - t) * v_a1 + t * v_a2
    #             v_b = (1 - u) * v_b1 + u * v_b2

    #             v_rel = v_a - v_b
    #             v_normal_mag = v_rel.dot(normal)

    #             # only resolve if objects are moving towards each other
    #             if v_normal_mag < 0.01:
    #                 w_a = (1 - t)**2 * inv_mass_a1 + t**2 * inv_mass_a2
    #                 w_b = (1 - u)**2 * inv_mass_b1 + u**2 * inv_mass_b2
    #                 total_inv_mass = w_a + w_b

    #                 if total_inv_mass > gs.EPS:
    #                     restitution = (self.vertices_info[idx_a1].restitution + self.vertices_info[idx_b1].restitution) * 0.5
    #                     friction_coeff = (self.vertices_info[idx_a1].mu_k + self.vertices_info[idx_b1].mu_k) * 0.5

    #                     # calculate collision impulse (normal component) ---
    #                     delta_v_normal = -v_normal_mag * (1.0 + restitution)
    #                     jn = delta_v_normal / total_inv_mass
    #                     J_collision = jn * normal

    #                     # calculate friction impulse (tangential component)
    #                     v_tangent = v_rel - v_normal_mag * normal
    #                     v_tangent_norm = v_tangent.norm(gs.EPS)
    #                     J_friction = ti.Vector([0.0, 0.0, 0.0])

    #                     if v_tangent_norm > gs.EPS:
    #                         jt_required = v_tangent_norm / total_inv_mass
    #                         jt_friction = ti.min(jt_required, friction_coeff * jn)
    #                         J_friction = -v_tangent.normalized() * jt_friction

    #                     # combine impulses and distribute to the 4 vertices ---
    #                     J_total = J_collision + J_friction

    #                     ti.atomic_add(self.vertices[f, idx_a1, i_b].vel, J_total * inv_mass_a1 * (1 - t))
    #                     ti.atomic_add(self.vertices[f, idx_a2, i_b].vel, J_total * inv_mass_a2 * t)
    #                     ti.atomic_add(self.vertices[f, idx_b1, i_b].vel, -J_total * inv_mass_b1 * (1 - u))
    #                     ti.atomic_add(self.vertices[f, idx_b2, i_b].vel, -J_total * inv_mass_b2 * u)
