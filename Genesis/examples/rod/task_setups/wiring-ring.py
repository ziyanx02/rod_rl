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
            res=(1200, 900), pos=(-1.6, 1.0, 1.4), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(1200, 900), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
            lookat=(0.2, 0., 0), fov=20, GUI=False
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
            # color=(0.4, 1.0, 0.4),
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        )
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.008,
            static_friction=0.1,
            kinetic_friction=0.08,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=24,
            radius=0.04,
            axis="y",
            pos=(0.27, 0.0, 0.008),
            euler=(-30, 0, 0),
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
            segment_radius=0.008,
            static_friction=0.1,
            kinetic_friction=0.08,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=24,
            radius=0.04,
            axis="y",
            pos=(0.09, -0.27, 0.008),
            euler=(-30, 0, 90),
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
            pos=(0.45, -0.6, 0),
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
            pos=(0.7, 0.25, 0),
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

    x1 = 0.42
    x2 = 0.6
    x_delta = -0.1
    x1_move_forward = -0.13
    x2_move_forward = -0.05
    x2_move_forward2 = -0.17
    x2_move_forward3 = -0.03
    y_delta = -0.05
    y_delta2 = -0.135
    z = 0.013
    z_delta = 0.013
    z_delta2 = 0.013
    force = -3

    open_gap = 0.03

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0.0, z), initial_gripper_gap=open_gap)
    c1.set_initial_position()
    c2 = RobotController(scene, franka2, ef2, args, (x2, 0.0, z), initial_gripper_gap=open_gap)
    c2.set_initial_position()

    frames = defaultdict(list)

    # grasp
    c1.control_robot(0, 0)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    kids = r1.get_kinematic_indices()
    print("rod kids", kids)

    # lift
    c1.control_robot(0, 0, dz=z_delta)
    c2.control_robot(0, 0, dz=z_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    # move
    c1.control_robot(0, 0, dx=x_delta)
    c2.control_robot(0, 0, dx=x_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    # release f1, keep f2 close
    c1.control_robot(open_gap, open_gap)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # f1 fetch position, keep f2 close
    c1.control_robot(open_gap, open_gap, dz=0.1)
    c2.control_robot(0, 0, dx=x2_move_forward)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    
    # f1 fetch position, keep f2 close
    c1.control_robot(open_gap, open_gap, dx=x1_move_forward)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # f1 move down, keep f2 close
    c1.control_robot(open_gap, open_gap, dz=-0.1-z_delta2)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # f1 grasp
    c1.control_robot(force, force, g_dof_use_force=True)
    c2.control_robot(0, 0)
    for i in range(160):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    
    # f1 lift
    c1.control_robot(force, force, g_dof_use_force=True, dz=z_delta)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # move both
    c1.control_robot(0, 0, dx=0.5*x_delta)
    c2.control_robot(0, 0, dx=0.5*x_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    
    kids = r1.get_kinematic_indices()
    print("rod kids", kids)

    # release f2
    c1.control_robot(0, 0)
    c2.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # rotate f1 release and lift f2
    c1.control_robot(0, 0, dk=-30, degrees=True)
    c2.control_robot(open_gap*2, open_gap*2, dz=0.1)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(0, 0, dk=-30, degrees=True)
    c2.control_robot(open_gap*2, open_gap*2)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # move f1
    c1.control_robot(0, 0, dx=y_delta/np.sqrt(3), dy=y_delta)
    c2.control_robot(open_gap*2, open_gap*2, dx=y_delta/np.sqrt(3), dy=y_delta)
    for i in range(60):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(0, 0, dx=y_delta*0.95/np.sqrt(3), dy=y_delta)
    c2.control_robot(open_gap*2, open_gap*2, dx=y_delta*0.95/np.sqrt(3), dy=y_delta)
    for i in range(60):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # rotate f1 release f2
    c1.control_robot(0, 0, dk=-30, degrees=True)
    c2.control_robot(open_gap*2, open_gap*2, dk=-45, degrees=True)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # move f1
    c1.control_robot(force, force, g_dof_use_force=True, dy=y_delta2 / 2)
    c2.control_robot(open_gap*2, open_gap*2, dx=x2_move_forward2 / 2)
    for i in range(60):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # move f1
    c1.control_robot(force, force, g_dof_use_force=True, dy=y_delta2 / 2)
    c2.control_robot(open_gap*2, open_gap*2, dx=x2_move_forward2 / 2)
    for i in range(60):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and down f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(open_gap*2, open_gap*2, dz=-0.1-z_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release and lift f1 and grasp f2
    c1.control_robot(open_gap*2, open_gap*2, dz=0.2)
    c2.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    kids = r1.get_kinematic_indices()
    print("rod kids", kids)

    # release and offset f1 and lift f2
    c1.control_robot(open_gap*2, open_gap*2, dx=-1.5*x_delta-x1_move_forward)
    c2.control_robot(0, 0, dz=z_delta)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and move f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(0, 0, dx=y_delta/2, dy=y_delta/2)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and rotate f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(0, 0, dk=-45, degrees=True)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and move f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(0, 0, dx=x2_move_forward3, dy=y_delta/2)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(open_gap*2, open_gap*2)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    # release f1 and f2
    c1.control_robot(open_gap*2, open_gap*2)
    c2.control_robot(open_gap*2, open_gap*2, dz=0.1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    for i in range(60):
        print(i, scene.rod_solver.vertices[0, i, 0].vert)

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)


if __name__ == "__main__":
    main()
