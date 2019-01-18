import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon for a2C (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha for a2c (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training CPU processes to use (default: 2)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps (default: 20)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=16,
                        help='number of batches for ppo (default: 16)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='log interval, one log per n updates (default: 50)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='save interval, one save per n updates (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100) - Not used')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy - not implemented yet')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='saves and visualizes average reward over time')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--state-rep', default='sign',
                        help='state representation used : full, original, or sign (default: sign)')
    parser.add_argument('--env-name', default='TrafficLight-v0',
                        help='environment name, see gym_trafficlight Readme for list (default:TrafficLight-v0)')
    parser.add_argument('--penetration-type', default='constant',
                        help='penetration rate of environment during training, constant or linear (default:constant)')
    parser.add_argument('--no-log-waiting-time', action='store_true', default=False,
                        help='disables logging of env waiting times')
    parser.add_argument('--reward-type', default='local',
                        help='type of reward with regards to penetration : local, parial, global (default: local)') 
    parser.add_argument('--penetration-rate', type=float, default=1,
                        help='percentage of detected vehicles (default: 1)') 
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
