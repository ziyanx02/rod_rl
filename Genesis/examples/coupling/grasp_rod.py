import argparse
import mediapy
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug",backend=gs.gpu)

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
            damping=10.0,
            angular_damping=5.0,
        ),
        show_viewer=args.vis,
    )

    if args.path is not None:
        camera = scene.add_camera(
            res=(600, 450), pos=(3, -1, 1.5), up=(0, 0, 1),
            lookat=(0.65, 0., 0), fov=30, GUI=False
        )

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    # r1 = scene.add_entity(
    #     material=gs.materials.ROD.Base(
    #         segment_radius=0.01,
    #         segment_mass=0.001,
    #         E=1e5,
    #         G=1e4,
    #         plastic_yield=np.inf,
    #     ),
    #     morph=gs.morphs.ParameterizedRod(
    #         type="rod",
    #         n_vertices=20,
    #         interval=0.02,
    #         axis="x",
    #         pos=(0.465, 0.0, 0.02),
    #         euler=(0, 0, 0),
    #     ),
    #     surface=gs.surfaces.Default(
    #         color=(0.4, 1.0, 0.4),
    #         vis_mode='recon',
    #     )
    # )

    c1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=0.001,
            K=1e5,
            E=1e5,
            G=1e4,
            plastic_yield=np.inf,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=80,
            radius=0.16,
            axis="x",
            pos=(0.65, -0.16, 0.02),
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

    franka = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            # pos=(7, 3.5, 0.28),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

    gripper_geom_indices = list()
    lf = franka.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    if args.n_envs == 0:
        franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
    else:
        franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 50, 50]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
        np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
    )

    # end_effector = franka.get_link("hand")
    end_effector = franka.get_link("panda_grasptarget")

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.012]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.012]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    qpos[..., -2:] = 0.02

    franka.set_dofs_position(
        qpos
    )

    frames = list()

    # grasp
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-1, -1]) if args.n_envs == 0 else np.array([[-1, -1]] * args.n_envs), fingers_dof
    )  # can also use force control
    # franka.control_dofs_position(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # you can use position control
    for i in range(100):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    gs.logger.info("grasped")

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.22]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    # franka.control_dofs_position(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # you can use position control
    franka.control_dofs_force(
        np.array([-2, -2]) if args.n_envs == 0 else np.array([[-2, -2]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(80):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)
    
    # move
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.75, 0.0, 0.22]) if args.n_envs == 0 else np.array([[0.75, 0.0, 0.22]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    # franka.control_dofs_position(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # you can use position control
    franka.control_dofs_force(
        np.array([-2, -2]) if args.n_envs == 0 else np.array([[-2, -2]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(120):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    if args.path is not None:
        mediapy.write_video(args.path, np.array(frames), fps=30)


if __name__ == "__main__":
    main()
