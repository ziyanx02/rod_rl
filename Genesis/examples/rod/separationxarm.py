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
            substeps=10,
            # gravity=(0.,0.,0.)
        ),
        rod_options=gs.options.RodOptions(
            damping=25.0,
            angular_damping=10.0,
            n_pbd_iters=20,
        ),
        show_viewer=args.vis,
    )

    cameras = list()
    if args.path is not None:
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(0.5, 1.5, 1.), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=30, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(0.5, -1.5, 1.), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=30, GUI=False
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
        morph=gs.morphs.Rod(
            file="genesis/assets/meshes/ropec.npy",
            rest_state="straight",
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
        morph=gs.morphs.Rod(
            file="genesis/assets/meshes/roped.npy",
            rest_state="straight",
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
            pos=(0.7, -0.3, 0),
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
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 30, 30]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
            np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
        )

    ef1 = franka1.get_link("link_tcp")

    x1 = 0.54
    x_delta1 = -0.11
    y_delta1 = 0.087
    z = 0.1
    z_delta1 = -0.1
    z_delta_lift1 = 0.02
    z_delta_lift2 = 0.1

    # open_gap = 0.08
    open_gap = 0.015
    close_gap = 0.044

    force = 1

    debug = True

    frames = defaultdict(list)
    positions = list()

    print('red', scene.rod_solver.vertices.vert.to_numpy()[0,:50, 0])
    red_init_pos = scene.rod_solver.vertices.vert.to_numpy()[0,:50, :].copy()
    green_init_pos = scene.rod_solver.vertices.vert.to_numpy()[0,50:, :].copy()

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0, z), initial_gripper_gap=open_gap, debug=debug)
    c1.set_initial_position()
    p = c1.robot.get_dofs_position().cpu().numpy()
    positions.append(p)

    c1.control_robot(open_gap, open_gap, dk=-25)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(open_gap, open_gap, dy=y_delta1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(open_gap, open_gap, dz=z_delta1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True, dz=z_delta_lift1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True, dz=z_delta_lift2)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True, dx=-x_delta1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True, dx=-x_delta1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    c1.control_robot(force, force, g_dof_use_force=True, dx=-x_delta1)
    for i in range(80):
        scene.step()
        p = c1.robot.get_dofs_position().cpu().numpy()
        positions.append(p)
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

    positions = np.array(positions).squeeze(1)
    print(positions.shape)
    np.savez(args.path.replace(".mp4", f"_pos.npz"),
             positions=positions,
             red_init_pos=red_init_pos,
             green_init_pos=green_init_pos)

if __name__ == "__main__":
    main()