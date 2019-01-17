import copy
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import update_linear_schedule, get_time, save_params
from a2c_ppo_acktr.visualize import visualize

args = get_args()

## Check if algorithm args are correct
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

## Number of epochs / updates  -----  num_steps is num of episodes before update
num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

# Set observation_space shapes
occ_obs_shape = (4,125)
sign_obs_shape = 10

## Initialize all seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## If CPU is used
if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Generate save_path
save_path = ""
if args.save_dir != "":
    save_path = os.path.join(args.save_dir, args.algo, get_time())
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    save_params(args, os.path.join(save_path, "parameters.txt"))
args.save_path = save_path


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    ## Make environments
    envs = make_vec_envs(args, device)

    ## Setup Policy / network architecture
    actor_critic = Policy(occ_obs_shape, sign_obs_shape, args.state_rep, envs.action_space, args.recurrent_policy)
    actor_critic.to(device)

    ## Choose algorithm to use
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    ## Initiate memory buffer
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        occ_obs_shape, sign_obs_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    ## Start env with first observation
    occ_obs, sign_obs = envs.reset()
    if args.state_rep == 'full':
        rollouts.occ_obs[0].copy_(occ_obs)
    rollouts.sign_obs[0].copy_(sign_obs)
    rollouts.to(device)

    # Last 20 rewards - can set different queue length for different averaging
    episode_rewards = deque(maxlen=args.num_steps)
    reward_track = []
    start = time.time()

    ## Loop over every policy update
    for j in range(num_updates):

        ## Setup parameter decays
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        ## Loop over num_steps environment updates to form trajectory
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # Pass observation through network and get outputs
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.occ_obs[step], rollouts.sign_obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Do action in environment and save reward
            occ_obs, sign_obs, reward, done, _ = envs.step(action)
            episode_rewards.append(reward.numpy())

            # Masks the processes which are done
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            # Insert step information in buffer
            rollouts.insert(occ_obs, sign_obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        ## Get state value of current env state
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.occ_obs[-1], rollouts.sign_obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        ## Computes the num_step return (next_value approximates reward after num_step) see Supp Material of https://arxiv.org/pdf/1804.02717.pdf
        ## Can use Generalized Advantage Estimation
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        # Update the policy with the rollouts
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        # Clean the rollout by cylcing last elements to first ones
        rollouts.after_update()

        ## Save model after each save_interval
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            torch.save(save_model, os.path.join(save_path, "model.pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if args.vis:
            # Add the average reward of update to reward tracker
            reward_track.append(np.mean(episode_rewards))

        ## Log progress
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy))

        ## Evaluate model on new environments for 10 rewards
        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(args, device, no_logging = True)

            eval_episode_rewards = []

            occ_obs, sign_obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        occ_obs, sign_obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                occ_obs, sign_obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])

                eval_episode_rewards.append(reward)

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

    ## Visualize tracked rewards(over num_steps) over time
    if args.vis:
        visualize(reward_track, args.algo, save_path)


if __name__ == "__main__":
    main()
