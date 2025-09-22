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
            # gravity=(0.,0.,-1.0)
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
            res=(600, 450), pos=(2, -1.4, 1.5), up=(0, 0, 1),
            lookat=(0.12, 0.2, 0.18), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1.5, -1.4, 1.4), up=(0, 0, 1),
            lookat=(0.12, 0.25, 0.18), fov=24, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.01,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.005
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.0005,
            K=9e5,  # 5e5
            E=4e5,
            G=0,
            plastic_yield=np.inf,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=24,
            interval=0.01,
            axis="x",
            pos=(0.0, 0.0, 0.21),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    b1 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=False
        ),
        morph=gs.morphs.Cylinder(
            radius=0.015,
            height=0.3,
            pos=(0, 0, 0.15),
            euler=(0, 0, 0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    b2 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=False
        ),
        morph=gs.morphs.Cylinder(
            radius=0.015,
            height=0.3,
            pos=(0.24, 0, 0.15),
            euler=(0, 0, 0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    sphere = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, rho=200, coup_friction=0.02,
        ),
        morph=gs.morphs.Sphere(
            radius=0.02,
            pos=(0.12, 0.06, 0.2),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 1.0)
        )
    )

    cube = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, rho=20, coup_friction=0.02,
        ),
        morph=gs.morphs.Box(
            pos=(0.12, 0.23, 0.22),
            size=(0.08, 0.08, 0.08),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.7, 0.7, 1.0)
        )
    )

    table = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.02,
        ),
        morph=gs.morphs.Box(
            pos=(0.12, 1.0, 0.09),
            size=(0.8, 1.9, 0.18),
            euler=(0, 0, 0),
            fixed=True,
        ),
    )

    fks = list()

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=1.0
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/xarm/xarm7_with_gripper_reduced_dof.urdf',
            pos=(-0.25, -0.65, 0),
            euler=(0., 0., 90.),
            fixed=True,
            collision=True,
            links_to_keep=['link_tcp'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )
    fks.append(franka1)

    gripper_geom_indices = list()
    lf = franka1.get_link("left_finger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka1.get_link("right_finger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)

    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    r1.set_fixed_states(
        fixed_ids=[0, 1, 21, 22]
    )

    # Optional: set control gains
    for f in [franka1]:
        if args.n_envs == 0:
            f.set_qpos(np.array([0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0]))
        else:
            f.set_qpos(np.array([[0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 300, 300]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 30, 30]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -300, -300]),
            np.array([87, 87, 87, 87, 12, 12, 12, 300, 300]),
        )

    ef1 = franka1.get_link("link_tcp")

    x1 = 0.12
    z = 0.218
    y = -0.1
    y_delta = 0.11
    y_back = -0.11

    force1 = 1.5
    force2 = 1.5

    open_gap = 0.02
    close_gap = 0.044

    debug = True

    frames = defaultdict(list)
    positions = list()

    r1_init_pos = scene.rod_solver.vertices.vert.to_numpy()[0,:24, :].copy()

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, y, z), initial_gripper_gap=open_gap, debug=debug)
    c1.set_initial_position()
    p = c1.robot.get_dofs_position().cpu().numpy()
    positions.append(p)

    c1.control_robot(open_gap, open_gap, di=90)
    for i in range(200):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    c1.control_robot(open_gap, open_gap, dy=y_delta)
    for i in range(120):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("fetch")

    c1.control_robot(force1, force1, g_dof_use_force=True)
    for i in range(160):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    # stretch slingshot
    c1.control_robot(force2, force2, g_dof_use_force=True, dy=y_back)
    for i in range(20):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("stretched")

    # release slingshot
    c1.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released")

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

    positions = np.array(positions).squeeze(1)
    print(positions.shape)
    np.savez(args.path.replace(".mp4", "_pos.npz"),
             positions=positions,
             init_pos=r1_init_pos)

if __name__ == "__main__":
    main()
