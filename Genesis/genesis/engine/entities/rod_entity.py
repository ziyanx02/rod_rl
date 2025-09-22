import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import RODEntityState
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING, to_gs_tensor, tensor_to_array

from .base_entity import Entity


@ti.data_oriented
class RodEntity(Entity):
    """
    A discrete linear object (DLO)-based entity for rod simulation.

    This class represents a deformable object using tetrahedral elements. It interfaces with
    the physics solver to handle state updates, checkpointing, gradients, and actuation
    for physics-based simulation in batched environments.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : Solver
        The physics solver instance used for simulation.
    material : Material
        The material properties defining elasticity, density, etc.
    morph : Morph
        The morph specification that defines the entity's shape.
    surface : Surface
        The surface mesh associated with the entity (for rendering or collision).
    idx : int
        Unique identifier of the entity within the scene.
    rod_idx : int, optional
        Index of this rod in the solver (default is 0).
    v_start : int, optional
        Starting index of this entity's vertices in the global vertex array (default is 0).
    e_start : int, optional
        Starting index of this entity's edges in the global edge array (default is 0).
    iv_start : int, optional
        Starting index of this entity's internal vertices in the global internal vertex array (default is 0).
    visualize_twist : bool, optional
        Whether to visualize twist frames applied to this (Rod) entity as arrows in the viewer and rendered images. Note that this will not be displayed in images rendered by camera using the `RayTracer` renderer.
    """

    def __init__(
        self, scene, solver, material, morph, surface, idx, 
        rod_idx=0, v_start=0, e_start=0, iv_start=0, visualize_twist=False
    ):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._rod_idx = rod_idx     # index of this rod in the solver
        self._v_start = v_start     # offset for vertex index
        self._e_start = e_start     # offset for edge index
        self._iv_start = iv_start   # offset for internal vertex index
        self._step_global_added = None
        self._visualize_twist = visualize_twist

        self._surface.update_texture()

        self.sample()

        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos_single(self, pos, verts_idx):
        for i_b in range(self._sim._B):
            i_global = verts_idx + self._v_start
            for j in ti.static(range(3)):
                self.solver.vertices[0, i_global, i_b].vert[j] = pos[i_b, j]

    
    def set_position(self, pos):
        """
        Set the target position(s) for the Rod entity.

        Parameters
        ----------
        pos : torch.Tensor or array-like
            The desired position(s). Can be:
            - (3,): a single COM offset vector.
            - (n_vertices, 3): per-vertex positions for all vertices.
            - (n_envs, 3): per-environment COM offsets.
            - (n_envs, n_vertices, 3): full batched per-vertex positions.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self.init_positions_COM_offset + pos
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self.n_vertices, 3):
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self.init_positions_COM_offset.unsqueeze(0) + pos.unsqueeze(1)
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_velocity(self, vel):
        """
        Set the target velocity(ies) for the Rod entity.

        Parameters
        ----------
        vel : torch.Tensor or array-like
            The desired velocity(ies). Can be:
            - (3,): a global velocity vector for all vertices.
            - (n_vertices, 3): per-vertex velocities.
            - (n_envs, 3): per-environment velocities broadcast to all vertices.
            - (n_envs, n_vertices, 3): full batched per-vertex velocities.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self.n_vertices, 3):
                self._tgt["vel"] = vel.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel.unsqueeze(1).tile((1, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def get_state(self):
        state = RODEntityState(self, self._sim.cur_step_global)
        self.get_frame(
            self._sim.cur_substep_local,
            state.pos,
            state.vel,
            state.fixed,
            state.theta,
            state.omega,
        )

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def get_kinematic_indices(self):
        # TODO: env idx
        kinematic_indices = list()
        f = self._sim.cur_substep_local
        for i_v in range(self.n_vertices):
            i_global = self._v_start + i_v
            if self.solver.vertices_ng[f, i_v, 0].is_kinematic:
                kinematic_indices.append(i_global)
        return kinematic_indices

    # def deactivate(self):         # NOTE: Not used
    #     gs.logger.info(f"{self.__class__.__name__} <{self.id}> deactivated.")
    #     self._tgt["act"] = gs.INACTIVE
    #     self.active = False

    # def activate(self):           # NOTE: Not used
    #     gs.logger.info(f"{self.__class__.__name__} <{self.id}> activated.")
    #     self._tgt["act"] = gs.ACTIVE
    #     self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, rest_state):
        """
        Initialize Rod entity with given vertices.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        Raises
        ------
        Exception
            If no vertices are provided.
        """
        verts = verts.astype(gs.np_float, copy=False)
        n_verts = verts.shape[0]

        # rotate
        R = gu.quat_to_R(np.array(self.morph.quat, dtype=gs.np_float))
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM

        if not init_positions.shape[0] > 0:
            gs.raise_exception(f"Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.init_positions_COM_offset = self.init_positions - gs.tensor(verts_COM)

        edges = list()
        is_loop = self.is_loop
        n_edges = n_verts if is_loop else n_verts - 1
        for i in range(n_edges):
            # NOTE: check this
            edges.append(verts[(i + 1) % n_verts] - verts[i])
        edges = np.array(edges, dtype=gs.np_float)

        self.edges = gs.tensor(edges)

        # resolve rest state
        if rest_state == "default":
            self.rest_positions = self.init_positions.clone()
        elif rest_state == "straight":
            # definitely not loop
            rest_positions = np.zeros_like(init_positions)
            lengths = np.linalg.norm(edges, axis=1)
            # create a straight line along x axis
            for i in range(n_edges):
                rest_positions[i + 1][0] = rest_positions[i][0] + lengths[i]
            self.rest_positions = gs.tensor(rest_positions)
            gs.logger.info(
                f"Entity {self.uid}({self._rod_idx}) initialized with rest state 'straight', "
                f"min_el: {lengths.min():.2e}, max_el: {lengths.max():.2e}, mean_el: {lengths.mean():.2e}."
            )
        else:
            gs.raise_exception(f"Unsupported rest state {rest_state}.")

    def _sample_rod(self, n_vertices: int, interval: float, axis: int):
        verts = list()
        for i in range(n_vertices):
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = i * interval
            verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def _sample_circle(self, n_vertices: int, radius: float, axis: int, gap: int):
        verts = list()
        for i in range(n_vertices):
            theta = 2 * np.pi * i / (n_vertices + gap)     # +1 to avoid overlap at the end
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = radius * np.cos(theta)
            vert[(axis + 1) % 3] = radius * np.sin(theta)
            verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def _sample_half_circle(self, n_vertices: int, radius: float, axis: int, gap: int):
        verts = list()
        for i in range(n_vertices + 2 * gap):
            theta = np.pi * i / (n_vertices + 2 * gap - 1)  # Adjusted to cover half circle
            vert = np.zeros(3, dtype=np.float64)
            vert[axis] = radius * np.cos(theta)
            vert[(axis + 1) % 3] = radius * np.sin(theta)
            if gap <= i < n_vertices + gap:
                verts.append(vert.reshape(3))
        verts = np.stack(verts, axis=0)
        return verts

    def sample(self):
        """
        Sample mesh and elements based on the entity's morph type.

        Raises
        ------
        Exception
            If the morph type is unsupported.
        """

        file_path = getattr(self.morph, 'file', None)
        if file_path is None:
            # Parametric morph
            if self.morph.axis == "x":
                axis = 0
            elif self.morph.axis == "y":
                axis = 1
            elif self.morph.axis == "z":
                axis = 2
            else:
                gs.raise_exception(f"Unsupported axis {self.morph.axis}.")

            if self.morph.type == "rod":
                vertices = self._sample_rod(self.morph.n_vertices, self.morph.interval, axis)
            elif self.morph.type == "circle":
                vertices = self._sample_circle(self.morph.n_vertices, self.morph.radius, axis, self.morph.gap)
            elif self.morph.type == "half_circle":
                vertices = self._sample_half_circle(self.morph.n_vertices, self.morph.radius, axis, self.morph.gap)
            else:
                gs.raise_exception(f"Unsupported morph type {self.morph.type}.")
            vertices = vertices + self.morph.pos
        else:
            vertices = np.load(self.morph.file)
            assert vertices.ndim == 2, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
            assert vertices.shape[1] == 3, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
            vertices = vertices + self.morph.pos

        self.instantiate(vertices, self.morph.rest_state)

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid}({self._rod_idx}) added. class: {self.__class__.__name__}, "
                f"morph: {self.morph.__class__.__name__}, #v: {self.n_vertices}, "
                f"o: {self.is_loop}, fix: {self.morph.fixed}, material: {self.material}."
            )

        # Convert to appropriate numpy array types
        verts_np = tensor_to_array(self.init_positions, dtype=gs.np_float)
        rest_verts_np = tensor_to_array(self.rest_positions, dtype=gs.np_float)
        edges_np = tensor_to_array(self.edges, dtype=gs.np_float)

        self._solver._kernel_add_rods(
            rod_idx=self._rod_idx,
            is_loop=self.is_loop,
            use_inextensible=self.material.use_inextensible,
            stretching_stiffness=self.material.K,
            bending_stiffness=self.material.E,
            twisting_stiffness=self.material.G,
            plastic_yield=self.material.plastic_yield,
            plastic_creep=self.material.plastic_creep,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            n_verts=self.n_vertices,
        )

        self._solver._kernel_finalize_rest_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            segment_mass=self.material.segment_mass,
            segment_radius=self.material.segment_radius,
            static_friction=self.material.static_friction,
            kinetic_friction=self.material.kinetic_friction,
            restitution=self.material.restitution,
            verts_rest=rest_verts_np,
            edges_rest=edges_np,
        )

        self._solver._kernel_finalize_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            fixed=self.morph.fixed,
            verts=verts_np,
            edges=edges_np,
        )
        self.active = True

    # ------------------------------------------------------------------------------------
    # ---------------------------- checkpoint and buffer ---------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_keys(self):
        """
        Initialize the keys used in target state management.

        This defines which physical properties (e.g., position, velocity) will be tracked for checkpointing and buffering.
        """
        self._tgt_keys = ["vel", "pos", "fixed", "omega", "theta"]

    def init_tgt_vars(self):
        """
        Initialize the target state variables and their buffers.

        This sets up internal dictionaries to store per-step target values for properties like velocity, position, actuation, and activation.
        """

        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        """
        Initialize checkpoint storage for simulation state.
        """
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        """
        Save the current target state buffers to a checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to save.
        """
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {
                "_tgt_buffer": dict(),
            }

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        """
        Restore target state buffers from a previously saved checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to load.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        """
        Clear target buffers and any externally queried simulation states.

        Used before backpropagation to reset gradients.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]

        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["vel"])

        if self._tgt["fixed"] is not None:
            self._tgt["fixed"].assert_contiguous()
            self._tgt["fixed"].assert_sceneless()
            self.set_fixed(self._sim.cur_substep_local, self._tgt["fixed"])

        if self._tgt["theta"] is not None:
            self._tgt["theta"].assert_contiguous()
            self._tgt["theta"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["theta"])

        if self._tgt["omega"] is not None:
            self._tgt["omega"].assert_contiguous()
            self._tgt["omega"].assert_sceneless()
            self.set_omega(self._sim.cur_substep_local, self._tgt["omega"])

        # clear kinematic states
        self._solver._kernel_clear_kinematic_states_all_substeps()
        # clear contact states
        self._solver._kernel_clear_contact_states_all_substeps()

        for key in self._tgt_keys:
            self._tgt[key] = None

    def process_input_grad(self):
        """
        Process gradients of input states and propagate them backward.

        Notes
        -----
        Automatically applies the backward hooks for position and velocity tensors.
        Clears the gradients in the solver to avoid double accumulation.
        """
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_pos = self._tgt_buffer["pos"].pop()
        _tgt_omega = self._tgt_buffer["omega"].pop()
        _tgt_theta = self._tgt_buffer["theta"].pop()

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

        if _tgt_omega is not None and _tgt_omega.requires_grad:
            _tgt_omega._backward_from_ti(self.set_omega_grad, self._sim.cur_substep_local)

        if _tgt_theta is not None and _tgt_theta.requires_grad:
            _tgt_theta._backward_from_ti(self.set_theta_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None or _tgt_pos is not None:
            # manually zero the grad since manually setting state breaks gradient flow
            self.clear_grad(self._sim.cur_substep_local)

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(self, f, pos):
        """
        Set vertex positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing new positions.
        """

        self._solver._kernel_set_vertices_pos(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos=pos,
        )

    def set_pos_grad(self, f, pos_grad):
        """
        Set gradient of vertex positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of positions.
        """

        self._solver._kernel_set_vertices_pos_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set vertex velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing velocities.
        """

        self._solver._kernel_set_vertices_vel(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradient of vertex velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of velocities.
        """

        self._solver._kernel_set_vertices_vel_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    def set_theta(self, f, theta):
        """
        Set edge twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        theta : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing twist angles.
        """

        self._solver._kernel_set_edges_theta(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega=theta,
        )

    def set_theta_grad(self, f, theta_grad):
        """
        Set gradient of edge twist angles (in radian) in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        theta_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing gradients of twist angles.
        """

        self._solver._kernel_set_edges_theta_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            theta_grad=theta_grad,
        )

    def set_omega(self, f, omega):
        """
        Set edge angular velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        omega : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing angular velocities.
        """

        self._solver._kernel_set_edges_omega(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega=omega,
        )

    def set_omega_grad(self, f, omega_grad):
        """
        Set gradient of edge angular velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        omega_grad : gs.Tensor
            Tensor of shape (n_envs, n_edges,) containing gradients of angular velocities.
        """

        self._solver._kernel_set_edges_omega_grad(
            f=f,
            e_start=self._e_start,
            n_edges=self.n_edges,
            omega_grad=omega_grad,
        )
    
    def set_fixed(self, f, fixed):
        """
        Set the fixed status of each vertex in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        fixed : gs.Tensor
            Tensor of shape (n_envs, n_vertices,) containing boolean fixed status for each vertex.
        """

        self._solver._kernel_set_fixed_states(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            fixed=fixed,
        )

    # def set_d1_ref(self, f, d1_ref):
    #     """
    #     Set the reference frame d1s for each edge in the solver.

    #     Parameters
    #     ----------
    #     f : int
    #         Current substep/frame index.

    #     d1_ref : gs.Tensor
    #         Tensor of shape (n_envs, n_edges, 3) containing reference frame d1s.
    #     """

    #     self._solver._kernel_set_edges_d1_ref(
    #         f=f,
    #         e_start=self._e_start,
    #         n_edges=self.n_edges,
    #         d1_ref=d1_ref,
    #     )
    
    # def set_d2_ref(self, f, d2_ref):
    #     """
    #     Set the reference frame d2s for each edge in the solver.

    #     Parameters
    #     ----------
    #     f : int
    #         Current substep/frame index.

    #     d2_ref : gs.Tensor
    #         Tensor of shape (n_envs, n_edges, 3) containing reference frame d2s.
    #     """

    #     self._solver._kernel_set_edges_d2_ref(
    #         f=f,
    #         e_start=self._e_start,
    #         n_edges=self.n_edges,
    #         d2_ref=d2_ref,
    #     )

    # def set_d3(self, f, d3):
    #     """
    #     Set the material frame tangents for each edge in the solver.

    #     Parameters
    #     ----------
    #     f : int
    #         Current substep/frame index.

    #     d3 : gs.Tensor
    #         Tensor of shape (n_envs, n_edges, 3) containing material frame tangents.
    #     """

    #     self._solver._kernel_set_edges_d3(
    #         f=f,
    #         e_start=self._e_start,
    #         n_edges=self.n_edges,
    #         d3=d3,
    #     )

    @gs.assert_built
    def set_init_vertices(self, verts_np, edges_np):
        self._solver._kernel_finalize_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            verts=verts_np,
            edges=edges_np,
        )

    @gs.assert_built
    def set_fixed_states(self, fixed_states=None, fixed_ids=None):
        """
        Set the fixed status of each vertex. This method is used to fixed vertices along the whole simulation
        before it starts.

        Parameters
        ----------
        fixed_states: list or np.ndarray
            List or array of booleans indicating fixed status for each vertex. Shape should be (n_vertices,).
        fixed_ids: list
            List of vertex indices to be fixed.
        """

        if fixed_ids is None and fixed_states is None:
            is_fixed = np.zeros(self.n_vertices, dtype=gs.np_bool)
        elif fixed_ids is None and fixed_states is not None:
            is_fixed = np.asarray(fixed_states).copy().reshape(-1).astype(gs.np_bool)
            assert is_fixed.shape[0] == self.n_vertices, \
                f"Fixed states has {is_fixed.shape[0]} vertices, but rod {self._rod_idx} has {self.n_vertices}."
        elif fixed_ids is not None:
            is_fixed = [1 if i in fixed_ids else 0 for i in range(self.n_vertices)]
            is_fixed = np.array(is_fixed, dtype=gs.np_bool)
        else:
            raise ValueError("`fixed_ids` and `fixed_states` cannot be provided at the same time.")

        is_fixed = np.tile(is_fixed, (self._sim._B, 1))  # (n_envs, n_vertices)

        # set fixed states for the first local frame
        self._solver._kernel_set_fixed_states(
            f=0,    # start from 0, then f -> f+1
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            fixed=is_fixed,
        )

    @ti.kernel
    def _kernel_get_verts_pos(self, f: ti.i32, pos: ti.types.ndarray(), verts_idx: ti.types.ndarray()):
        # get current position of vertices
        for i_v, i_b in ti.ndrange(verts_idx.shape[0], verts_idx.shape[1]):
            i_global = verts_idx[i_v, i_b] + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]

    @ti.kernel
    def get_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        fixed: ti.types.ndarray(),
        theta: ti.types.ndarray(),
        omega: ti.types.ndarray(),
    ):
        """
        Extract the state of particles at the given frame.

        Parameters
        ----------
        f : int
            The substep/frame index to fetch the state from.

        pos : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store positions.

        vel : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store velocities.

        fixed : np.ndarray
            Output array of shape (n_envs, n_vertices) to store fixed status.

        theta : np.ndarray
            Output array of shape (n_envs, n_edges) to store twist angles.

        omega : np.ndarray
            Output array of shape (n_envs, n_edges) to store angular velocities.
        """

        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]
                vel[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vel[j]
            fixed[i_b, i_v] = self._solver.vertices_ng[f, i_global, i_b].fixed

        for i_e, i_b in ti.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            theta[i_b, i_e] = self._solver.edges[f, i_global, i_b].theta
            omega[i_b, i_e] = self._solver.edges[f, i_global, i_b].omega

    @ti.kernel
    def get_all_verts_kernel(
        self,
        pos: ti.types.ndarray(),
    ):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[0, i_global, i_b].vert[j]

    def get_all_verts(self):
        base_v_shape = (self.sim._B, self.n_vertices, 3)
        args = {
            "dtype": gs.np_float,
            # "requires_grad": False,
            # "scene": self.scene,
        }
        pos = np.zeros(base_v_shape, **args)
        self.get_all_verts_kernel(pos)
        return pos

    @ti.kernel
    def set_frame_add_grad_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                self._solver.vertices.grad[f, i_global, i_b].vert[j] += pos_grad[i_b, i_v, j]

    @ti.kernel
    def set_frame_add_grad_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                self._solver.vertices.grad[f, i_global, i_b].vel[j] += vel_grad[i_b, i_v, j]

    @ti.kernel
    def set_frame_add_grad_theta(self, f: ti.i32, theta_grad: ti.types.ndarray()):
        for i_e, i_b in ti.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].theta += theta_grad[i_b, i_e]

    @ti.kernel
    def set_frame_add_grad_omega(self, f: ti.i32, omega_grad: ti.types.ndarray()):
        for i_e, i_b in ti.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].omega += omega_grad[i_b, i_e]
    
    def add_grad_from_state(self, state):
        """
        Accumulate gradients from a recorded state back into the solver.

        Parameters
        ----------
        state : RODEntityState
            The state object containing gradients for physical quantities.
        """
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self.set_frame_add_grad_pos(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self.set_frame_add_grad_vel(self._sim.cur_substep_local, state.vel.grad)

        if state.theta.grad is not None:
            state.theta.assert_contiguous()
            self.set_frame_add_grad_theta(self._sim.cur_substep_local, state.theta.grad)

        if state.omega.grad is not None:
            state.omega.assert_contiguous()
            self.set_frame_add_grad_omega(self._sim.cur_substep_local, state.omega.grad)

    @ti.kernel
    def clear_grad(self, f: ti.i32):
        """
        Zero out the gradients of position, velocity, and angular velocity for the current substep.

        Parameters
        ----------
        f : int
            The substep/frame index for which to clear gradients.

        Notes
        -----
        This method is primarily used during backward passes to manually reset gradients
        that may be corrupted by explicit state setting.
        """
        # TODO: not well-tested
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            self._solver.vertices.grad[f, i_global, i_b].vert = 0
            self._solver.vertices.grad[f, i_global, i_b].vel = 0
        for i_e, i_b in ti.ndrange(self.n_edges, self._sim._B):
            i_global = i_e + self.e_start
            self._solver.edges.grad[f, i_global, i_b].theta = 0
            self._solver.edges.grad[f, i_global, i_b].omega = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the Rod entity."""
        return len(self.init_positions)

    @property
    def n_edges(self):
        """Number of edges in the Rod entity."""
        return len(self.edges)

    @property
    def is_loop(self):
        """Whether the rod is loop."""
        return self.morph.is_loop

    @property
    def n_internal_vertices(self):
        """Number of internal vertices in the Rod entity."""
        return len(self.init_positions) if self.is_loop else len(self.init_positions) - 2

    @property
    def n_dofs(self):
        """Number of degrees of freedom (DOFs) in the Rod entity."""
        # 3 for each vertex + 1 for each edge
        return 3 * self.n_vertices + self.n_edges

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def e_start(self):
        """Global edge index offset for this entity."""
        return self._e_start

    @property
    def iv_start(self):
        """Global internal vertex index offset for this entity."""
        return self._iv_start

    @property
    def visualize_twist(self):
        """Whether to visualize twist frames."""
        return self._visualize_twist

    @property
    def material(self):
        """Material properties of the Rod entity."""
        return self._material
