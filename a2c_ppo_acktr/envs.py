import os

import gym
import numpy as np
import torch
import gym_trafficlight

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym_trafficlight.wrappers.visualization_wrapper import  TrafficVisualizationWrapper
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper

def make_env(seed, rank, state_rep, allow_early_resets,visual):
    def _thunk():

        env = gym.make('TrafficLight-v0')
        env.seed(seed + rank)

        env = TrafficParameterSetWrapper(env, {"state_representation" : state_rep}).unwrapped

        if visual:
            # env = TrafficParameterSetWrapper(env, args).unwrapped (to use of params are to pass down to env creation)
            env = TrafficVisualizationWrapper(env).unwrapped
        return env

    return _thunk

# Make a vector of environments (one for each process)
def make_vec_envs(seed, num_processes,state_rep,
                  device, allow_early_resets, num_frame_stack=None,visual=False):

    envs = [make_env(seed, i, state_rep,allow_early_resets,visual)
            for i in range(num_processes)]

    # Choose wrapper
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Choose another wrapper
    envs = VecPyTorch(envs, state_rep, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, state_rep, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.state_rep = state_rep

    def _convert_raw_obs(self, obs):

        obs = np.squeeze(obs,axis=1)
        if self.state_rep == "full":
            occ_obs = np.asarray([np.asarray(occ) for occ in obs[:,0]])
            occ_obs = torch.from_numpy(occ_obs).float().to(self.device)

            sign_obs = np.asarray([np.asarray(sign) for sign in obs[:,1]])
            sign_obs = torch.from_numpy(sign_obs).float().to(self.device)

            return occ_obs , sign_obs
        else :
            obs = torch.from_numpy(obs).float().to(self.device)
            return None , obs

    def reset(self):

        obs = self.venv.reset()
        return self._convert_raw_obs(obs)        

    def step_async(self, actions):
        #actions = actions.squeeze(1).cpu().numpy()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if reward.ndim > 1:
            reward = np.squeeze(reward, axis=1)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        occ_obs, sign_obs = self._convert_raw_obs(obs)
        return occ_obs, sign_obs, reward, done, info

