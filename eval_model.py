import argparse
import numpy as np
import torch

from a2c_ppo_acktr.envs import make_vec_envs


num_processes = 1
seed = 1
device = "cpu"

def main():

    args = get_args()

    actor_critic = torch.load(args.model_path)
        

    eval_envs = make_vec_envs(seed + num_processes, num_processes, device, True, visual = True)

    eval_episode_rewards = []

    obs = eval_envs.reset()

    # This is used for RNN, not fully implemented yet
    eval_recurrent_hidden_states = torch.zeros(num_processes,
                    actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < args.num_steps:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])

        eval_episode_rewards.append(reward)

    eval_envs.close()

    print(" Mean reward after {} steps: {:.5f}\n".
        format(len(eval_episode_rewards),
               np.mean(eval_episode_rewards)))

def get_args():

    parser = argparse.ArgumentParser(description='eval_RL')
    parser.add_argument('--model_path', default='./trained_models/a2c/TrafficLight-v0.pt',
                        help='path of the saved model to evaluate (default: ./trained_models/a2c/TrafficLight-v0.pt')
    parser.add_argument('--num_steps', type=int, default=500,
                        help='number of environment steps to run (default: 500)')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()