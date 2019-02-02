#!/bin/bash

cd ..

python3 main.py --algo ppo --penetration-rate 1 --env-name TrafficLight-Lust12408-regular-time-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full

python3 main.py --algo ppo --penetration-rate 0.1 --env-name TrafficLight-Lust12408-regular-time-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full

#Full LuxMedium pen 1 + 0.1 PPO