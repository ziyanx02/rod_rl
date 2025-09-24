import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import TorchUtils

from train_env_wiring_ring import Train_Env_Wiring_ring
from train_env_wiring_post import Train_Env_Wiring_post
from train_env_coiling import Train_Env_Coiling
from train_env_slingshot import Train_Env_Slingshot
from train_env_knotting import Train_Env_Knotting
from train_env_gathering import Train_Env_Gathering
from train_env_separation import Train_Env_Separation
from train_env_wireart import Train_Env_Wireart
from train_env_wrapping import Train_Env_Wrapping
from train_env_lifting import Train_Env_Lifting

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], n_features[2])
        self._h4 = nn.Linear(n_features[2], n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))

        # Ensure parameters are float32 to avoid Float/Double mismatches
        self.float()

    def forward(self, state, **kwargs):
        x = torch.squeeze(state, 1)
        # Align dtype/device with layer weights to avoid mismatches
        x = x.to(dtype=self._h1.weight.dtype, device=self._h1.weight.device)
        features1 = F.relu(self._h1(x))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = self._h4(features3)

        return a

def experiment(alg, n_envs,n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params, env_name="wiring_ring"):

    logger = Logger(alg.__name__ + "_1_legged_gym", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    if env_name == "wiring_ring":
        mdp = Train_Env_Wiring_ring(n_envs=n_envs, GUI=False)
    elif env_name == "wiring_post":
        mdp = Train_Env_Wiring_post(n_envs=n_envs, GUI=False)
    elif env_name == "coiling":
        mdp = Train_Env_Coiling(n_envs=n_envs, GUI=False)
    elif env_name == "slingshot":
        mdp = Train_Env_Slingshot(n_envs=n_envs, GUI=False)
    elif env_name == "knotting":
        mdp = Train_Env_Knotting(n_envs=n_envs, GUI=False)
    elif env_name == "gathering":
        mdp = Train_Env_Gathering(n_envs=n_envs, GUI=False)
    elif env_name == "separation":
        mdp = Train_Env_Separation(n_envs=n_envs, GUI=False)
    elif env_name == "wireart":
        mdp = Train_Env_Wireart(n_envs=n_envs, GUI=False)
    elif env_name == "wrapping":
        mdp = Train_Env_Wrapping(n_envs=n_envs, GUI=False)
    elif env_name == "lifting":
        mdp = Train_Env_Lifting(n_envs=n_envs, GUI=True)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    # Prepare curve logging file: logs/curve/<env_name>/<EXP_ID>
    def _get_min_unused_exp_id(directory: Path) -> int:
        existing_ids = set()
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file() and child.suffix == ".txt" and child.stem.isdigit():
                    existing_ids.add(int(child.stem))
        exp_id = 0
        while exp_id in existing_ids:
            exp_id += 1
        return exp_id

    curve_dir = Path("logs") / "curve" / env_name
    curve_dir.mkdir(parents=True, exist_ok=True)
    exp_id = _get_min_unused_exp_id(curve_dir)
    curve_path = curve_dir / f"{exp_id}.txt"
    curve_file = open(curve_path, "w")

    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((n_envs*24) / 16),
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = VectorCore(agent, mdp)
    
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy().to("cpu").item()
    V = torch.mean(agent._V(dataset.get_init_states())).cpu().item()

    logger.epoch_info(0, J=J, R=R, entropy=E, V=V)
    print("Starting training")
    for it in trange(n_epochs, leave=False):
        print(it)
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        if (it + 1) % 5 == 0 or it == n_epochs - 1:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)
        else:
            dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy().to("cpu").item()
        V = torch.mean(agent._V(dataset.get_init_states())).cpu().item()

        print(R)

        logger.epoch_info(it+1, J=J, R=R, entropy=E, V=V)

        # Log reward for this iteration to curve file
        curve_file.write(f"{it+1},{R}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())

        del dataset

    # Close curve file after training
    curve_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='wiring_ring', help='Environment name: wiring_ring or wiring_post')
    args = parser.parse_args()

    # Enforce float32 globally for new tensors/modules
    torch.set_default_dtype(torch.float32)
    TorchUtils.set_default_device('cuda:0')
    n_envs = 2
    ppo_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': 1e-3}},
        n_epochs_policy=5,
        batch_size=int((n_envs*24) / 16),
        eps_ppo=.2,
        lam=.95,
        ent_coeff=0.01
    )
    policy_params = dict(
        std_0=1.,
        n_features=[512, 256, 128],
        use_cuda=True
    )
    experiment(alg=PPO, n_envs=n_envs, n_epochs=40, n_steps=n_envs*24*50, n_steps_per_fit=n_envs*24,
        n_episodes_test=n_envs, alg_params=ppo_params, policy_params=policy_params, env_name=args.env_name)
