import genesis as gs
import imageio
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os 
import json
import matplotlib.pyplot as plt


class Train_Env():
    def __init__(self, task, scene=None, GUI=False, log_dir=None, n_envs=None):
        self.task = task
        self.GUI = GUI
        self.n_envs = n_envs
        gs.init(seed=0, precision="64", logging_level="INFO", backend=gs.gpu, performance_mode=True)
        if scene is None:
            viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            )

            self.scene = gs.Scene(
                viewer_options=viewer_options,
                sim_options=gs.options.SimOptions(
                    dt=1e-3,
                    substeps=5,
                    # gravity=(0.,0.,0.)
                ),
                rod_options=gs.options.RodOptions(
                    damping=15.0,
                    angular_damping=10.0,
                    n_pbd_iters=20,
                ),
                show_viewer=self.GUI,
            )
        else:
            self.scene = scene

        self.scene_built_for_training = False
        self.scene_built_for_evaluation = False
        self.img_save_dir = None
        self.img_steps = 0

        self.cmaes_optimizer_created = False
        self.iter = 0

        self.create_log_dir(log_dir)

        self.construct_scene()

        
    def create_log_dir(self, log_dir):
        log_dir = os.path.join(log_dir, 'try')
        os.makedirs(log_dir, exist_ok=True)
        n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
        self.img_save_dir = os.path.join(log_dir, f"{n_tries:03d}")
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.img_save_dir, "opt_log"), exist_ok=True)
        
    def init_mass(self, mass=0.015):
        for entity in self.scene.sim.rigid_solver.entities[2:]:
            for link in entity.links:
                link._inertial_mass = mass
    
    def construct_scene(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()
        self.img_steps += 1

    def save_gif(self, save_dir):
        images = []
        file_list = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        file_list.sort()
        for f in file_list:
            images.append(imageio.imread(os.path.join(save_dir, f)))
        imageio.mimsave(os.path.join(save_dir, 'movie.gif'), images)
    
    def reset(self):
        self.scene.reset()


    