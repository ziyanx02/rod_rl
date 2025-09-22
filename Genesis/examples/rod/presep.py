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
            substeps=25,
            # gravity=(0.,0.,0.)
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
            E=5e3,
            G=0,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=30,
            interval=0.02,
            axis="x",
            pos=(0, 0.0, 0.04),
            euler=(0, 0, 15),
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
            E=5e3,
            G=0,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=30,
            interval=0.02,
            axis="x",
            pos=(0, 0.0, 0.02),
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
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.55, -0.4, 0),
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
    rf = franka1.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)

    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    # Optional: set control gains
    for f in [franka1]:
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

    x1 = 0.57
    z = 0.013
    z_delta1 = 0.053

    open_gap = 0.04

    # move to pre-grasp pose
    c1 = RobotController(scene, franka1, ef1, args, (x1, 0, z), initial_gripper_gap=open_gap)
    c1.set_initial_position()

    frames = defaultdict(list)

    # grasp
    c1.control_robot(0, 0)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("grasped")

    r1.set_fixed_states(
        fixed_ids=[0, 1, 28, 29]
    )
    r2.set_fixed_states(
        fixed_ids=[0, 1]
    )

    print('0', scene.rod_solver.vertices[0, 0, 0].vert)
    print('29', scene.rod_solver.vertices[0, 29, 0].vert)
    print('30', scene.rod_solver.vertices[0, 30, 0].vert)
    print('59', scene.rod_solver.vertices[0, 59, 0].vert)

    c1.control_robot(0, 0, dz=z_delta1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("lifted")

    c1.control_robot(0, 0, dx=-0.08, dy=0.2)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("moved")

    c1.control_robot(0, 0, dz=-z_delta1)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("down")

    r2.set_fixed_states(
        fixed_ids=[0, 1, 28, 29]
    )

    c1.control_robot(open_gap, open_gap)
    for i in range(80):
        scene.step()
        for cid, cam in enumerate(cameras):
            img = cam.render()[0]
            frames[cid].append(img)
    gs.logger.info("released")

    # export vertices
    vertices = scene.rod_solver.vertices.vert.to_numpy()[0, :, 0]
    print(vertices.shape)
    np.save("ropea.npy", vertices[:30])
    np.save("ropeb.npy", vertices[30:60])

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()