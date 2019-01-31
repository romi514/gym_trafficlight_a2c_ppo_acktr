import os

import gym
import numpy as np
import torch
import gym_trafficlight

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from gym_trafficlight.wrappers.visualization_wrapper import  TrafficVisualizationWrapper
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper
from gym_trafficlight import PenetrationRateManager


def make_env(args, rank, no_logging,visual):
    def _thunk():


        env = gym.make(args.env_name)
        env.seed(args.seed + rank)

        env_args = {}
        env_args['state_representation'] = args.state_rep
        env_args['reward_type'] = args.reward_type
        env_args['penetration_rate'] = args.penetration_rate

        if not no_logging and not args.no_log_waiting_time and rank == 0:
            if args.save_dir != "":
                env_args['log_waiting_time'] = True
                env_args['logger_type'] ='baselines_logger'
                #env_args['record_file'] = os.path.join(args.save_path, "waiting_time_process_"+str(rank+1)+".txt")
            else:
                print("No waiting time logging is done because no save file is given")

        if args.penetration_type == 'linear':
            prm = PenetrationRateManager(
                trend = 'linear',
                transition_time = 3*365, #3 years
                pr_start = 0.1,
                pr_end = 1
                )
            env_args['reset_manager'] = prm

        env = TrafficParameterSetWrapper(env, env_args).unwrapped

        if visual:
            env = TrafficVisualizationWrapper(env).unwrapped

        return env

    return _thunk

# Make a vector of environments (one for each process)
def make_vec_envs(args, device, no_logging = False, visual=False):

    envs = [make_env(args, i, no_logging,visual)
            for i in range(args.num_processes)]

    envs = SubprocVecEnv(envs)

    # Choose another wrapper
    envs = VecPyTorch(envs, args.state_rep, device)

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

