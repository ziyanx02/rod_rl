import argparse
import mediapy
import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('./examples/rod')
from controller import RobotController
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)

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
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
        ),
        rod_options=gs.options.RodOptions(
            damping=5.0,
            angular_damping=5.0,
            n_pbd_iters=20,
        ),
        show_viewer=args.vis,
    )

    cameras = list()
    if args.path is not None:
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1.2, 0.8, 1.0), up=(0, 0, 1),
            lookat=(0.6, 0.3, 0), fov=30, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-0.2, -1.5, 0.6), up=(0, 0, 1),
            lookat=(0.45, 0.3, 0), fov=30, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.1,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            E=1e4,
            G=1e3,
            plastic_yield=0.2,
            plastic_creep=0.9,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=45,
            interval=0.02,
            axis="x",
            pos=(-0.04, 0.0, 0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.006,
            static_friction=0.1,
            kinetic_friction=0.08,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=16,
            radius=0.032,
            axis="y",
            pos=(0.28, 0.0, 0.006),
            euler=(-15, 0, 0),
            gap=1,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        )
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.006,
            static_friction=0.1,
            kinetic_friction=0.08,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=16,
            radius=0.032,
            axis="y",
            pos=(0.56, 0.0, 0.006),
            euler=(-15, 0, 0),
            gap=1,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        )
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.7
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.61, -0.36, 0),
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
            pos=(0.6, 0.6, 0),
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

    # end_effector = franka.get_link("hand")
    ef1 = franka1.get_link("panda_grasptarget")
    ef2 = franka2.get_link("panda_grasptarget")

    x1 = -0.02
    x2 = 0.82
    z = 0.013
    z_delta = 0.013
    force = -0.5
    force2 = -1

    open_gap = 0.04

    # initial grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0.0, z))
    c1.set_initial_position()
    c2 = RobotController(scene, franka2, ef2, args, (x2, 0.0, z))
    c2.set_initial_position()

    frames = defaultdict(list)

    # 1. grasp and lift
    c1.control_robot(0, 0)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")
    c1.control_robot(0, 0, dz=z_delta)
    c2.control_robot(0, 0, dz=z_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    kids = r1.get_kinematic_indices()
    print("rod kids", kids)

    # 2. rotate
    c1.control_robot(0, 0)
    c2.rotate_around_point(
        0, 0,
        center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=-45, pos_angle=45,
    )
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated")

    # 3. rotate
    c1.control_robot(0, 0)
    c2.rotate_around_point(
        0, 0,
        center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=-45, pos_angle=45,
    )
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated")

    # 5. rotate f1
    # c1.control_robot(0, 0)
    # c2.rotate_around_point(
    #     0, 0,
    #     center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
    #     axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
    #     angle=-10, pos_angle=10,
    #     rot_mask=[False, False, True]
    # )
    # for i in range(80):
    #     scene.step()
    #     for cid, cam in enumerate(cameras):
    #         img = cam.render()[0]
    #         frames[cid].append(img)
    # gs.logger.info("rotated")

    # 5. rotate f1 f2 + 20
    # c1.control_robot(0, 0)
    # c2.control_robot(0, 0)
    # for i in range(40):
    #     scene.step()
    #     for cid, cam in enumerate(cameras):
    #         img = cam.render()[0]
    #         frames[cid].append(img)

    c1.control_robot(0, 0)
    c2.rotate_around_point(
        0, 0,
        center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=-10, pos_angle=10,
        rot_mask=[False, False, True]
    )
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated")

    # 5. rotate f1 f2 + 30
    c1.control_robot(0.015, 0.015, dz=-z_delta)
    c2.rotate_around_point(
        0, 0,
        center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=-10, pos_angle=10,
        rot_mask=[False, False, True]
    )
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated")

    # 5. rotate f1 f2 + 40
    c1.control_robot(0, 0)
    c2.rotate_around_point(
        0, 0,
        center=torch.tensor([0.565, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=-15, pos_angle=15,
        rot_mask=[False, False, True]
    )
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasp f1")

    # 5. rotate f1 f2 + 50
    c1.control_robot(0, 0)
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasp f1 release f2")

    # 5. lift f2
    c1.control_robot(0, 0, dz=z_delta)
    c2.control_robot(open_gap, open_gap, dx=0.25, dz=0.15)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("close f1")

    # 5. lift f2
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=30, pos_angle=-30,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lift f1")

    # 5. rotate f1 - 30
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=30, pos_angle=-30,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 5. rotate f1 - 60
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=30, pos_angle=-30,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    c1.control_robot(0, 0)
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 5. rotate f1 - 90
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=10, pos_angle=-10,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 5. rotate f1 - 100
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=10, pos_angle=-10,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 5. rotate f1 - 110
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.275, 0.0, z+z_delta], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=10, pos_angle=-10,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 5. rotate f1 - 120
    c1.control_robot(0, 0, dx=0.02, dy=0.02, rot_mask=[False, False, True])
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated f1")

    # 6. release f1
    c1.control_robot(0.03, 0.03, rot_mask=[False, False, True])
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released f1")

    # 6. release f1
    c1.control_robot(open_gap, open_gap, dz=0.1, rot_mask=[False, False, True])
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released f1")

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()