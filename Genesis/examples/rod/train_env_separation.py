import genesis as gs
import imageio
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os 
import json
import matplotlib.pyplot as plt
from train_env import Train_Env
from ring_crossing_helper import ring_crossing_count_axis_aligned, ring_center_from_axis_aligned_vertices, closest_distance_rope_to_point
from mushroom_rl.core import MDPInfo
from mushroom_rl.rl_utils.spaces import Box

class Train_Env_Separation(Train_Env):
    def __init__(self, task='wiring', log_dir="xxx/wiring", n_envs=5, GUI=False):
        # Control vertices on the first rope
        self.control_idx = [27]

        super().__init__(task, n_envs=n_envs, log_dir=log_dir, GUI=GUI)

        # Vectorized RL configuration (aligned with Train_Env_Wiring_ring)
        self._backend = 'numpy'

        # Observation / action specs
        # Use actual vertex counts from built scene (two ropes)
        self._obs_dim = (self.rope.n_vertices + self.rope2.n_vertices) * 3
        self._act_dim = len(self.control_idx) * 3
        self._horizon = 200
        self._steps_per_action = 10

        # Observation/action spaces
        low_obs = np.full((self._obs_dim,), -np.inf, dtype=np.float32)
        high_obs = np.full((self._obs_dim,), np.inf, dtype=np.float32)
        observation_space = Box(low_obs, high_obs)

        act_limit = 0.02
        low_act = -np.ones((self._act_dim,), dtype=np.float32) * act_limit
        high_act = np.ones((self._act_dim,), dtype=np.float32) * act_limit
        action_space = Box(low_act, high_act)

        # Control dt approximates sim dt * internal steps
        control_dt = self.scene.sim_options.dt * self._steps_per_action
        self._mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=self._horizon, dt=control_dt, backend=self._backend)

    def construct_scene(self):
        plane = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        segment_radius = 0.01
        self.rope = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                E=5e3,
                G=1e3,
            ),
            morph=gs.morphs.Rod(
                file="meshes/ropea.npy",
                rest_state="straight",
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        self.rope2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                E=5e3,
                G=0,
            ),
            morph=gs.morphs.Rod(
                file="meshes/ropeb.npy",
                rest_state="straight",
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 1.0, 0.4),
                vis_mode='recon',
            )
        )

        self.scene.rod_solver.register_gripper_geom_indices([])

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        # Fix control vertices across all envs for direct kinematic control
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        # Record initial control vertex positions for resets (per env)
        verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
        self._initial_ctrl_pos = verts_rope[:, self.control_idx].copy()  # (n_envs, n_ctrl, 3)

    def reward(self):
        A = self.rope.get_all_verts()
        B = self.rope2.get_all_verts()

        # Pairwise distances between ropes for each env:
        # D shape: (n_envs, n_verts_A, n_verts_B)
        diff = A[:, :, None, :] - B[:, None, :, :]
        D = np.linalg.norm(diff, axis=-1)

        # For each vertex in A, distance to nearest in B; and vice versa
        a_to_b_min = D.min(axis=2)  # (n_envs, n_verts_A)
        b_to_a_min = D.min(axis=1)  # (n_envs, n_verts_B)

        # Symmetric NN distance (Chamfer-style), averaged per env
        rewards = a_to_b_min.mean(axis=1) + b_to_a_min.mean(axis=1)  # (n_envs,)

        # Larger reward -> ropes farther apart
        return rewards.tolist()

    # ------------------------- MushroomRL Vectorized Environment API -------------------------
    @property
    def info(self):
        return self._mdp_info

    @property
    def number(self):
        return self.n_envs

    def _compute_observation(self):
        verts_a = self.rope.get_all_verts()   # (n_envs, n_vertices_a, 3)
        verts_b = self.rope2.get_all_verts()  # (n_envs, n_vertices_b, 3)
        obs = np.concatenate([
            verts_a.reshape(self.n_envs, -1),
            verts_b.reshape(self.n_envs, -1)
        ], axis=1).astype(np.float32)
        return obs

    def reset_all(self, env_mask, state=None):
        self.scene.reset()

        # Fix control vertices across all envs for direct kinematic control
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        self.scene.step()

        obs = self._compute_observation()
        return obs, [{}] * self.n_envs

    def step_all(self, env_mask, action):
        # Accept torch or numpy; operate and return numpy for numpy backend
        if isinstance(action, torch.Tensor):
            action = action.detach().to('cpu').numpy()
        else:
            action = np.asarray(action)
        if isinstance(env_mask, torch.Tensor):
            env_mask_np = env_mask.detach().to('cpu').numpy().astype(bool)
        else:
            env_mask_np = np.asarray(env_mask, dtype=bool)

        # Clip actions and reshape to (n_envs, n_ctrl, 3)
        action = action.astype(np.float32)
        action = np.clip(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        delta = action.reshape(self.n_envs, -1, 3)

        verts_rope = self.rope.get_all_verts()
        current_ctrl = verts_rope[:, self.control_idx]  # (n_envs, n_ctrl, 3)

        # Zero-out actions for non-masked envs
        masked_delta = np.zeros_like(delta, dtype=np.float32)
        masked_delta[env_mask_np] = delta[env_mask_np]

        # Track failure states and absorbing flags (only track masked envs)
        absorbing = np.zeros((self.n_envs,), dtype=bool)
        tracked = env_mask_np.copy()
        alive = tracked.copy()

        # Pre-step NaN detection before any micro-step of this macro-step
        nan_now = np.isnan(verts_rope).any(axis=(1, 2))
        newly_nan = nan_now & alive
        if newly_nan.any():
            absorbing[newly_nan] = True
            alive[newly_nan] = False

        for j in range(self._steps_per_action):
            if not (alive & tracked).any():
                break

            interp = (j + 1) / self._steps_per_action
            target_ctrl = current_ctrl + masked_delta * interp
            target_ctrl[:, :, 2] = np.clip(target_ctrl[:, :, 2], 0.01, None)
            for k, vi in enumerate(self.control_idx):
                self.rope.set_pos_single(target_ctrl[:, k], vi)
            self.scene.step()

            # Collision detection on control vertices (first rope)
            collided = self.rope._solver.vertices_ng.is_collided.to_numpy()  # (n_envs, n_vertices)
            verts_to_check = np.array(self.control_idx) + self.rope._v_start
            collided_ctrl = collided[:, verts_to_check].any(axis=1)

            newly_collided = collided_ctrl & alive
            if newly_collided.any():
                absorbing[newly_collided] = True
                alive[newly_collided] = False

            # NaN detection after stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                absorbing[newly_nan_after] = True
                alive[newly_nan_after] = False

        next_obs = self._compute_observation()
        rewards = np.array(self.reward(), dtype=np.float32)

        return next_obs, rewards, absorbing, [{}] * self.n_envs

    def render_all(self, env_mask, record=False):
        if self.GUI:
            pass
        if record:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return None
    
    def step(self, actions):
        raise NotImplementedError()
        # to be done

    def eval_traj(self, trajs):
        """
        Evaluate trajectories.

        Rewards:
        - If an env survives all micro-steps: reward = self.reward()[env].
        - If an env COLLIDES or gets NaNs in verts: reward = survival_time / total_micro_steps.
        - If env reward is NaN at the end: reward = -100.

        Survival time counts micro-steps from 0..N, where N = n_steps * steps_interval.
        """
        import numpy as np

        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 3 == 0 and dof // 3 == n_ctrl, (
            f"dof must be 3 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        steps_interval = 200
        total_micro_steps = int(n_steps * steps_interval)
        if total_micro_steps <= 0:
            # Degenerate case: no steps → everyone "survives"; defer to env reward (or -100 if NaN)
            rewards = np.asarray(self.reward(), dtype=np.float32)
            rewards[np.isnan(rewards)] = -100.0
            return rewards.astype(np.float32)

        # Per-env status
        alive = np.ones((self.n_envs,), dtype=bool)              # True until first failure (collision or NaN)
        ever_nan = np.zeros((self.n_envs,), dtype=bool)          # True if verts ever became NaN
        ever_collided = np.zeros((self.n_envs,), dtype=bool)     # True if collision occurred
        first_fail_step = np.full((self.n_envs,), total_micro_steps, dtype=np.int32)  # micro-step index of first failure

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            newly_nan = nan_now & alive
            if newly_nan.any():
                # Failure occurs before any micro-step of this macro-step
                # Use step = max(1, i*steps_interval) to keep survival count >= 1 if we want strictly positive
                step_at_nan = i * steps_interval
                step_at_nan = max(1, step_at_nan)
                first_fail_step[newly_nan] = step_at_nan
                ever_nan[newly_nan] = True
                alive[newly_nan] = False

            # Early exit if everyone is already NaN
            if ever_nan.all():
                break

            # If no env is alive anymore, we can stop
            if not alive.any():
                break

            # Prepare interpolation to targets for this macro-step
            current_pos = verts_rope[:, self.control_idx]              # (n_envs, n_ctrl, 3)
            delta = trajs[:, i].reshape(self.n_envs, -1, 3)            # (n_envs, n_ctrl, 3)

            for j in range(steps_interval):
                if not alive.any():
                    break

                alpha = (j + 1) / steps_interval
                target_pos = current_pos + delta * alpha               # (n_envs, n_ctrl, 3)

                # Apply target positions; if set_pos_single isn't batch-aware, loop envs instead.
                for k in range(n_ctrl):
                    self.rope.set_pos_single(target_pos[:, k], self.control_idx[k])

                self.scene.step()

                # Post-step: detect collisions
                collided = self.rope._solver.vertices_ng.is_collided.to_numpy()  # (n_envs, n_vertices)
                verts_to_check = np.array(self.control_idx) + self.rope._v_start
                collided_ctrl = collided[:, verts_to_check].any(axis=1)          # (n_envs,)

                newly_collided = collided_ctrl & alive
                if newly_collided.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_collided] = np.minimum(first_fail_step[newly_collided], global_step)
                    ever_collided[newly_collided] = True
                    alive[newly_collided] = False

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_nan_after] = np.minimum(first_fail_step[newly_nan_after], global_step)
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        final = np.empty((n_envs,), dtype=np.float32)

        failed = ~alive  # failed due to collision or NaN during rollout
        survived = alive

        # Failed: reward = survival_ratio (counts both collision and NaN cases)
        if failed.any():
            survival_ratio = first_fail_step.astype(np.float32) / float(total_micro_steps)
            final[failed] = survival_ratio[failed]

        # Survived full rollout: take env reward; if it's NaN, clamp to -100
        final[survived] = env_rewards[survived]
        if env_rewards_nan.any():
            final[env_rewards_nan] = -100.0

        return final.astype(np.float32)


        
