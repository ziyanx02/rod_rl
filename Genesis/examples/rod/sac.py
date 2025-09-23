import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
MUSHROOM_PATH = os.path.join(REPO_ROOT, 'mushroom-rl')
if MUSHROOM_PATH not in sys.path:
    sys.path.append(MUSHROOM_PATH)

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.utils import TorchUtils

from train_env_wiring_ring import Train_Env_Wiring_ring
from train_env_wiring_post import Train_Env_Wiring_post
from train_env_coiling import Train_Env_Coiling
from train_env_slingshot import Train_Env_Slingshot
from train_env_knotting import Train_Env_Knotting
from train_env_gathering import Train_Env_Gathering
from train_env_separation import Train_Env_Separation
from train_env_wireart import Train_Env_Wireart
from train_env_lifting import Train_Env_Lifting

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

def experiment(alg, n_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, env_name="wiring_ring"):
    logger = Logger(alg.__name__, results_dir='./logs', log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP selection
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
    elif env_name == "lifting":
        mdp = Train_Env_Lifting(n_envs=n_envs, GUI=False)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    # Genesis env initialization may set torch default dtype to float64;
    # re-enforce float32 before building networks/agent to avoid dtype mismatch.
    torch.set_default_dtype(torch.float32)

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    # Approximators
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    # Core
    core = VectorCore(agent, mdp)

    # Initial evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(dataset.state).item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    # Warmup
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # Training
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state).item()

        logger.epoch_info(n+1, J=J, R=R, entropy=E)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='wiring_ring',
                        help='Environment name: wiring_ring, wiring_post, coiling, slingshot, knotting, gathering, separation, wireart')
    parser.add_argument('--n_envs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    # Enforce float32 globally for new tensors/modules
    torch.set_default_dtype(torch.float32)
    TorchUtils.set_default_device('cuda:0')

    n_envs = args.n_envs
    experiment(
        alg=SAC,
        n_envs=n_envs,
        n_epochs=args.epochs,
        n_steps=n_envs*24*50,
        n_steps_per_fit=n_envs*24,
        n_episodes_test=256,
        env_name=args.env_name
    )
