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
            res=(600, 450), pos=(3, -1, 1.5), up=(0, 0, 1),
            lookat=(0.65, 0., 0), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
            lookat=(0.2, 0., 0), fov=30, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.1,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.01
    l1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            K=5e5,
            E=1e4,
            G=0,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=50,
            radius=0.14,
            axis="x",
            pos=(0.65, 0, 0.02),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=1.0
    )

    c1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.Cylinder(
            radius=0.143,
            height=0.2,
            pos=(0.1, 0., 0.1),
            fixed=True,
        ),
    )

    fks = list()

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.1, -0.5, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )
    fks.append(franka1)

    franka2 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.1, 0.5, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )
    fks.append(franka2)


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

    print('length', scene.rod_solver.edges_info[0].length_rest)

    # r1.set_fixed_states(
    #     fixed_ids=[0, 1, 43, 44]
    # )

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    for f in fks:
        if args.n_envs == 0:
            f.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
        else:
            f.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 30, 30]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
            np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
        )

    # end_effector = franka.get_link("hand")
    ef1 = franka1.get_link("panda_grasptarget")
    ef2 = franka2.get_link("panda_grasptarget")

    x1 = 0.65
    y1 = -0.14
    y1_delta = 0.04
    x2 = 0.65
    y2 = 0.14
    y2_delta = -0.04
    x_delta = -0.3
    z = 0.014
    z_delta = 0.15
    z_delta2 = 0.05
    force = -2

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, y1, z), initial_gripper_gap=0.04)
    c1.set_initial_position()
    c2 = RobotController(scene, franka2, ef2, args, (x2, y2, z), initial_gripper_gap=0.04)
    c2.set_initial_position()

    frames = defaultdict(list)
    # control_robot_abs(args, franka1, ef1, force, force, g_dof_use_force=True, x=x1, y=y1, z=z)
    # control_robot_abs(args, franka2, ef2, force, force, g_dof_use_force=True, x=x2, y=y2, z=z)
    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(force, force, g_dof_use_force=True)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    # control_robot_abs(args, franka1, ef1, force, force, g_dof_use_force=True, x=x1, y=y1+y1_delta, z=z+z_delta)
    # control_robot_abs(args, franka2, ef2, force, force, g_dof_use_force=True, x=x2, y=y2+y2_delta, z=z+z_delta)
    c1.control_robot(force, force, g_dof_use_force=True, dy=y1_delta, dz=z_delta)
    c2.control_robot(force, force, g_dof_use_force=True, dy=y2_delta, dz=z_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    kids = l1.get_kinematic_indices()
    print("rod kids", kids)

    # do1 = np.array([90, 0, 0])
    # quat1 = gu.xyz_to_quat(
    #     do1, rpy=True, degrees=True
    # )
    # tq1 = gu.transform_quat_by_quat(
    #     quat1, ef1.get_quat().cpu().numpy().reshape(-1)
    # )
    c1.control_robot(force, force, g_dof_use_force=True, di=90)
    c2.control_robot(force, force, g_dof_use_force=True)
    # control_robot_abs(args, franka1, ef1, force, force, g_dof_use_force=True, x=x1, y=y1+y1_delta, z=z+z_delta, quat=tq1)
    # control_robot_abs(args, franka2, ef2, force, force, g_dof_use_force=True, x=x2, y=y2+y2_delta, z=z+z_delta)
    for i in range(200):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("horizontal")

    # control_robot_abs(args, franka1, ef1, force, force, g_dof_use_force=True, x=x1+x_delta, y=y1+y1_delta, z=z+z_delta+z_delta2, quat=tq1)
    # control_robot_abs(args, franka2, ef2, force, force, g_dof_use_force=True, x=x2+x_delta, y=y2+y2_delta, z=z+z_delta+z_delta2)
    c1.control_robot(force, force, g_dof_use_force=True, dx=x_delta, dz=z_delta2)
    c2.control_robot(force, force, g_dof_use_force=True, dx=x_delta, dz=z_delta2)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")


    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()
