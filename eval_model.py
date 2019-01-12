
import numpy as np
import torch

from a2c_ppo_acktr.envs import make_vec_envs


save_path = "./trained_models/a2c/TrafficLight-v0-best.pt"
num_processes = 1
seed = 1
gamma = 0.99
device = "cpu"
num_iter = 500

def main():

    actor_critic = torch.load(save_path)
    eval_envs = make_vec_envs(seed + num_processes, num_processes, gamma, device, True)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes,
                    actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_iter:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])

        eval_episode_rewards.append(reward)

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
        format(len(eval_episode_rewards),
               np.mean(eval_episode_rewards)))

if __name__ == "__main__":
    main()