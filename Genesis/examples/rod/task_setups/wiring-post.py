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
            res=(600, 450), pos=(0.2, 1.2, 1.5), up=(0, 0, 1),
            lookat=(0.3, 0.2, 0), fov=30, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1.6, -1.2, 1.5), up=(0, 0, 1),
            lookat=(0.3, 0.2, 0), fov=30, GUI=False
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
            # K=1e5,
            E=1e3,
            G=1e3,
            # use_inextensible=False
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=60,
            interval=0.01,
            axis="x",
            pos=(0.3, 0.0, 0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=4,
            interval=0.02,
            axis="z",
            pos=(0.2, 0.1, -0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=4,
            interval=0.02,
            axis="z",
            pos=(0.1, 0.3, -0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.7
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.2, 0.45, 0),
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
            pos=(0.6, -0.3, 0),
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

    b1.set_fixed_states(
        fixed_ids=np.arange(10)
    )

    b2.set_fixed_states(
        fixed_ids=np.arange(10)
    )

    # Optional: set control gains
    for f in [franka1, franka2]:
        if args.n_envs == 0:
            f.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
        else:
            f.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
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

    x1 = 0.35
    x2 = 0.65
    x_delta = -0.05
    y_delta = 0.08
    z1 = 0.035
    z2 = 0.03
    z_down = -0.017
    z_delta1 = 0.008
    z_delta2 = 0.008
    force = -3

    open_gap = 0.03

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0.0, z1), initial_gripper_gap=open_gap)
    c1.set_initial_position()
    c2 = RobotController(scene, franka2, ef2, args, (x2, 0.0, z2), initial_gripper_gap=open_gap)
    c2.set_initial_position()

    frames = defaultdict(list)

    # grasp
    c1.control_robot(open_gap, open_gap, dz=z_down)
    c2.control_robot(open_gap, open_gap, dz=z_down)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("down")

    c1.control_robot(0, 0)
    c2.control_robot(0, 0)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    # lift
    c1.control_robot(0, 0, dz=z_delta1)
    c2.control_robot(0, 0, dz=z_delta2)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    kids = r1.get_kinematic_indices()
    print("rod kids", kids)

    # move
    for _ in range(6):
        c1.control_robot(0, 0, dx=x_delta)
        c2.control_robot(0, 0, dx=x_delta)
        for i in range(40):
            scene.step()
            for cid, cam in enumerate(cameras):
                img = cam.render()[0]
                frames[cid].append(img)
        gs.logger.info("moved")

    # move
    c1.control_robot(0, 0, dy=y_delta)
    c2.control_robot(0, 0, dy=y_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released 2")

    # rotate
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.2, 0.08, z1+z_down+z_delta1], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=45, pos_angle=-45,
        rot_mask=[False, False, True]
    )
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated 1")

    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fixed")

    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(open_gap, open_gap, dz=0.1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fixed")

    # rotate
    c1.rotate_around_point(
        0, 0,
        center=torch.tensor([0.2, 0.02, z1+z_down+z_delta1], dtype=gs.tc_float),
        axis=torch.tensor([0, 0, 1], dtype=gs.tc_float),
        angle=20, pos_angle=-20,
        rot_mask=[False, False, True]
    )
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("rotated 2")

    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(open_gap, open_gap, dz=0.1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fixed 1")

    c1.control_robot(0, 0, dy=y_delta)
    c2.control_robot(open_gap, open_gap, dz=0.1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fixed 2")

    c1.control_robot(0, 0, dy=y_delta)
    c2.control_robot(open_gap, open_gap, dz=0.1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fixed 3")


    for i in range(60):
        print(i, scene.rod_solver.vertices[0, i, 0].vert)

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)


if __name__ == "__main__":
    main()
