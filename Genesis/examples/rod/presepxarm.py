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
            substeps=10,
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
            needs_coup=True, coup_friction=0.3,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.005
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            E=4e4,
            G=1e3,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=50,
            interval=0.01,
            axis="x",
            pos=(0.1, 0.0, 0.04),
            euler=(0, 0, 24),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        )
    )

    r2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            E=4e4,
            G=1e3,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=50,
            interval=0.01,
            axis="x",
            pos=(0.1, 0.0, 0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.9
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/xarm/xarm7_with_gripper_reduced_dof.urdf',
            pos=(0.1, -0.4, 0),
            euler=(0., 0., 90.),
            fixed=True,
            collision=True,
            links_to_keep=['link_tcp'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

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

    # Optional: set control gains
    for f in [franka1]:
        if args.n_envs == 0:
            f.set_qpos(np.array([0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0]))
        else:
            f.set_qpos(np.array([[0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 30, 30]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -300, -300]),
            np.array([87, 87, 87, 87, 12, 12, 12, 300, 300]),
        )

    ef1 = franka1.get_link("link_tcp")

    x1 = 0.57
    z = 0.005
    z_delta1 = 0.01

    open_gap = 0.015
    close_gap = 0.044

    force1 = 1.5
    force2 = 1.5

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0, z), initial_gripper_gap=open_gap, debug=True)
    c1.set_initial_position()

    frames = defaultdict(list)

    # grasp
    c1.control_robot(force1, force1, g_dof_use_force=True)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    r1.set_fixed_states(
        fixed_ids=[0, 1, 48, 49]
    )
    # r2.set_fixed_states(
    #     fixed_ids=[0, 1]
    # )

    c1.control_robot(force2, force2, g_dof_use_force=True, dz=z_delta1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    c1.control_robot(force2, force2, g_dof_use_force=True, dz=2*z_delta1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    c1.control_robot(close_gap, close_gap, dx=-0.04, dy=0.05)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    c1.control_robot(close_gap, close_gap, dx=-0.04, dy=0.05)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    c1.control_robot(close_gap, close_gap, dy=0.05)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    c1.control_robot(close_gap, close_gap, dz=-z_delta1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("down")

    c1.control_robot(open_gap, open_gap)
    for i in range(120):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released")

    r2.set_fixed_states(
        fixed_ids=[0, 1, 48, 49]
    )

    # export vertices
    vertices = scene.rod_solver.vertices.vert.to_numpy()[0, :, 0]
    print(vertices.shape)
    np.save("ropec.npy", vertices[:50])
    np.save("roped.npy", vertices[50:100])

    print('0', scene.rod_solver.vertices[0, 0, 0].vert)
    print('43', scene.rod_solver.vertices[0, 43, 0].vert)
    print('45', scene.rod_solver.vertices[0, 45, 0].vert)
    print('47', scene.rod_solver.vertices[0, 47, 0].vert)
    print('49', scene.rod_solver.vertices[0, 49, 0].vert)
    print('50', scene.rod_solver.vertices[0, 50, 0].vert)
    print('93', scene.rod_solver.vertices[0, 93, 0].vert)
    print('95', scene.rod_solver.vertices[0, 95, 0].vert)
    print('97', scene.rod_solver.vertices[0, 97, 0].vert)
    print('99', scene.rod_solver.vertices[0, 59, 0].vert)

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()