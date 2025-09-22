import argparse
import mediapy
import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('./examples/rod')
from controller import RobotController
import genesis.utils.geom as gu
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug", backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=25,
            # gravity=(0.,0.,0.)
        ),
        rod_options=gs.options.RodOptions(
            damping=15.0,
            angular_damping=10.0,
            n_pbd_iters=20,
        ),
        show_viewer=args.vis,
    )

    cameras = list()
    if args.path is not None:
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(1.5, 1.7, 1.), up=(0, 0, 1),
            lookat=(0.4, 0., 0), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(0.2, 1.5, 1.), up=(0, 0, 1),
            lookat=(0.6, 0., 0), fov=20, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.01,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            E=1e4,
            G=0,
            # plastic_yield=np.inf,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=60,
            interval=0.02,
            axis="x",
            pos=(-0.3, 0.05, 0.5),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    cone1 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.3,
        ),
        morph=gs.morphs.Mesh(
            file="genesis/assets/meshes/cone.obj",
            pos=(0, 0, 0.25),
            scale=(7, 7, 5),
            # fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 1.0),
        ),
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.9
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(1.55, -0.4, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

    franka2 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(1.95, -0.4, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

    gripper_geom_indices = list()
    lf = franka1.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    lf = franka2.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka1.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka2.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)

    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    # Optional: set control gains
    for f in [franka1, franka2]:
        if args.n_envs == 0:
            f.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
        else:
            f.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
            np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
        )

    ef1 = franka1.get_link("panda_grasptarget")
    ef2 = franka2.get_link("panda_grasptarget")

    x1 = 0.57
    x2 = 0.75
    x_delta1 = 0.02
    x_delta2 = -0.02
    y = 0.12
    y_delta =-0.135
    z = 0.013
    z_delta1 = 0.053
    z_delta2 = 0.053
    z_delta_lift = 0.01

    open_gap = 0.04

    # move to pre-grasp pose
    # c1 = RobotController(scene, franka1, ef1, args, (x1, 0, z), initial_q_dof=open_gap)
    # c2 = RobotController(scene, franka2, ef2, args, (x2, y, z), initial_q_dof=open_gap)

    frames = defaultdict(list)

    # grasp
    # c1.control_robot(0, 0)
    # c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()