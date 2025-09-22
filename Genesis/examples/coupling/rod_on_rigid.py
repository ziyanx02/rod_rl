import argparse
import mediapy
import numpy as np
import genesis as gs
from collections import defaultdict


def test_v1(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            E=1e5,
            G=1e4
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=100,
            interval=0.01,
            axis="x",
            pos=(0.5, 0.5, 0.3),
            euler=(0.0, 0.0, 15.0),
        ),
        surface=gs.surfaces.Default(
            # diffuse_texture=gs.textures.ImageTexture(image_path="data/color.png",),   # Any image is OK.
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
        visualize_twist=True,       # Visualize the twist of the rod.
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            E=1e5,
            G=1e4
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=80,
            interval=0.01,
            axis="x",
            pos=(0.55, 0.43, 0.4),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="x",
            pos=(0.75, 0.435, 0.25),
            euler=(0.0, 0.0, -75.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="x",
            pos=(1.05, 0.435, 0.25),
            euler=(0.0, 0.0, -75.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=2)

    v1.set_fixed_states(
        fixed_ids = [0, 1]
    )
    v2.set_fixed_states(
        fixed_ids = [78, 79]
    )
    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )
    b2.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def test_v2(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_mass=0.1,
            segment_radius=0.005,
            E=1e6,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=80,
            interval=0.01,
            axis="x",
            pos=(0.75, 0.05, 0.35),
            euler=(0.0, 0.0, 15.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_mass=0.1,
            segment_radius=0.005,
            E=1e6,
            static_friction=1.5,
            kinetic_friction=1.25,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=80,
            interval=0.01,
            axis="x",
            pos=(0.75, 0.15, 0.35),
            euler=(0.0, 0.0, 15.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.025,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="x",
            pos=(1.0, 0.05, 0.3),
            euler=(0.0, 0.0, 105.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.025,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="x",
            pos=(1.25, 0.15, 0.3),
            euler=(0.0, 0.0, 105.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    # b3 = scene.add_entity(
    #     material=gs.materials.ROD.Base(
    #         segment_radius=0.025,
    #     ),
    #     morph=gs.morphs.ParameterizedRod(
    #         type="rod",
    #         n_vertices=3,
    #         interval=0.1,
    #         axis="x",
    #         pos=(0.85, 0.15, 0.3),
    #         euler=(0.0, 0.0, 105.0),
    #     ),
    #     surface=gs.surfaces.Default(
    #         color=(0.4, 0.4, 0.4),
    #         vis_mode='recon',
    #     ),
    # )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=2)

    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )
    b2.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )
    # b3.set_fixed_states(
    #     fixed_ids = [0, 1, 2]
    # )


def test_v3(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            E=1e6,
            G=1e4,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=120,
            radius=0.1,
            axis="x",
            pos=(1.02, -0.03, 0.22),
            euler=(0.0, 90.0, 105.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            E=5e5,
            G=5e5,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=20,
            interval=0.01,
            axis="x",
            pos=(0.92, 0.05, 0.12),
            euler=(0.0, 0.0, 15.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.015,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="x",
            pos=(0.9, 0.0, 0.07),
            euler=(0.0, 0.0, 105.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=2)

    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def test_v4(scene):
    E = 1e5
    G = 1e5
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.0075,
            E=E,
            G=G,
            static_friction=0.9,
            kinetic_friction=0.75
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=40,
            interval=0.01,
            axis="x",
            pos=(0.42, 0.42, 0.5),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.0075,
            E=E,
            G=G
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=20,
            interval=0.01,
            axis="x",
            pos=(0.42, 0.34, 0.54),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v3 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.0075,
            E=E,
            G=G,
            static_friction=1.5,
            kinetic_friction=1.25
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=60,
            interval=0.01,
            axis="x",
            pos=(0.42, 0.5, 0.46),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 1.0),
            vis_mode='recon',
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=2)


def test_v5(scene):
    E = 1e7
    G = 1e5
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.0075,
            E=E,
            G=G,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=40,
            interval=0.01,
            axis="x",
            pos=(0.3, 0.5, 0.5),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=2)


def test_v6(scene):
    E = 1e5
    G = 1e4
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            E=E,
            G=G,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=40,
            interval=0.02,
            axis="x",
            pos=(0.3, 0.5, 0.3),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=3,
            interval=0.1,
            axis="y",
            pos=(0.7, 0.4, 0.2),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)

    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def test_v7(scene):
    E = 1e5
    G = 1e4
    # Test initializing a half-circle rod with rest state "straight"
    # Better visualization with a frictionless plane as friction will
    # prevent the rod from springing back to its rest shape
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            E=E,
            G=G,
            segment_radius=0.01,
            static_friction=0.3,
            kinetic_friction=0.25,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="half_circle",
            n_vertices=30,
            radius=0.16,
            axis="z",
            pos=(0.28, 0.0, 0.03),
            euler=(90, 0, 0),
            rest_state="straight",
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        )
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--save_path", type=str, default=None)
    parser.add_argument("--fov", type=float, default=30)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("-st", "--substeps", type=int, default=20)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        rod_options=gs.options.RodOptions(
            damping=1.0,
            angular_damping=1.0,
            floor_height=0.0,
            adjacent_gap=2,
            n_pbd_iters=10
        ),
        show_viewer=args.vis,
    )

    if args.save_path is not None:
        cams = list()
        cams.append(scene.add_camera(
            res=(600, 450), pos=(2.6, 1.8, 1.6), up=(0, 0, 1),
            # res=(1024, 1024), pos=(1.6, 0.8, 0.6), up=(0, 0, 1),
            lookat=(0.9, 0.1, 0), fov=args.fov, GUI = False
        ))
        cams.append(scene.add_camera(
            res=(600, 450), pos=(1.2, 2.4, 1.6), up=(0, 0, 1),
            lookat=(0.5, 0.5, 0), fov=args.fov, GUI = False
        ))
    else:
        cam = None

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.0
    )
    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.3
    )

    plane = scene.add_entity(
        # material=frictionless_rigid,
        material=friction_rigid,
        morph=gs.morphs.Plane(),
    )

    # cube = scene.add_entity(
    #     material=frictionless_rigid,
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.5, 0.2),
    #         size=(0.2, 0.2, 0.2),
    #         euler=(30, 40, 0),
    #         fixed=True,
    #     ),
    # )

    # cube = scene.add_entity(
    #     material=frictionless_rigid,
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.5, 0.25),
    #         size=(0.2, 0.2, 0.2),
    #         euler=(0, 0, 0),
    #         # fixed=True,
    #     ),
    # )

    # sphere = scene.add_entity(
    #     # material=frictionless_rigid,
    #     material=friction_rigid,
    #     morph=gs.morphs.Sphere(
    #         radius=0.15,
    #         pos=(0.5, 0.48, 0.25),
    #         # fixed=True,
    #     )
    # )

    # test_v1(scene)
    # test_v2(scene)
    # test_v3(scene)
    # test_v4(scene)
    # test_v5(scene)
    # test_v6(scene)
    # test_v7(scene)

    frames = defaultdict(list)
    for i in range(args.steps):
        scene.step()
        for cid, cam in enumerate(cams):
            img = cam.render()[0]
            frames[cid].append(img)

    for cid in frames:
        mediapy.write_video(args.save_path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=10, qp=18)


if __name__ == "__main__":
    main()
