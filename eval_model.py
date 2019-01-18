import argparse
import numpy as np
import torch
import os

from argparse import Namespace

from a2c_ppo_acktr.envs import make_vec_envs


### CHANGE EVAL_MODEL -- NOT WORKING AFTER CHANGES

def main():

    args = get_args()


    actor_critic = torch.load(os.path.join(args.save_path,"model.pt"))
    params = load_params(os.path.join(args.save_path,"parameters.txt"))
    device = torch.device("cuda:0" if params.cuda else "cpu")

    eval_envs = make_vec_envs(params, device, visual = True)

    eval_episode_rewards = []

    occ_obs, sign_obs = eval_envs.reset()

    # This is used for RNN, not fully implemented yet
    eval_recurrent_hidden_states = torch.zeros(params.num_processes,
                    actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(params.num_processes, 1, device=device)

    while len(eval_episode_rewards) < args.num_steps:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                occ_obs, sign_obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        occ_obs, sign_obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])

        eval_episode_rewards.append(reward)

    eval_envs.close()

    print(" Mean reward after {} steps: {:.5f}\n".
        format(len(eval_episode_rewards),
               np.mean(eval_episode_rewards)))

def load_params(file_path):

    f = open(file_path,'r')
    params = Namespace()
    line = f.readline()
    while line:
        if '#' not in line:
            arg, str_value = tuple(line.split())
            if str_value == "True":
                value = True
                print("true")
            elif str_value == "False":
                value = False
                print("false")
            elif str_value == "None":
                value = None
                print("none")
            else: 
                try:
                    value = int(str_value)
                    print("int")
                except ValueError:
                    try:
                        value = float(str_value)
                        print("float")
                    except ValueError:
                        value = str_value
                        print("string")

            vars(params)[arg] = value
        line = f.readline()

    params.cuda = not params.no_cuda and torch.cuda.is_available()

    return params



def get_args():

    parser = argparse.ArgumentParser(description='eval_RL')
    parser.add_argument('--save-path', default='',
                        help='path of the saved model to evaluate (default: " ")')
    parser.add_argument('--num-steps', type=int, default=500,
                        help='number of environment steps to run (default: 500)')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()