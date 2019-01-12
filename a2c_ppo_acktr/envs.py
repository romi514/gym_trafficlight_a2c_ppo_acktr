import os

import gym
import numpy as np
import torch
import gym_trafficlight

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper


def make_env(seed, rank, allow_early_resets,visual):
    def _thunk():

        env = gym.make('TrafficLight-v0')
        env.seed(seed + rank)

        if visual:
            # env = TrafficParameterSetWrapper(env, args).unwrapped (to use of params are to pass down to env creation)
            env = TrafficVisualizationWrapper(env).unwrapped
        return env

    return _thunk

# Make a vector of environments (one for each process)
def make_vec_envs(seed, num_processes,
                  device, allow_early_resets, num_frame_stack=None,visual=False):

    envs = [make_env(seed, i, allow_early_resets,visual)
            for i in range(num_processes)]

    # Choose wrapper 
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Choose another wrapper
    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = np.squeeze(obs, axis=1)
        obs = torch.from_numpy(obs).float().to(self.device)

        print("RESET\n")
        print("RESET\n")
        print("RESET\n")
        print("RESET\n")
        print("RESET\n")
        print("RESET\n")



        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()        
        obs = np.squeeze(obs, axis=1)
        reward = np.squeeze(reward, axis=1)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
