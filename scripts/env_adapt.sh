#!/bin/bash

cd ..

python3 main.py --env-name TrafficLight-Lust12408-rush-hour-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 30000000 --penetration-rate 0.1 --state-rep full
wait
python3 main.py --env-name TrafficLight-Lust12408-rush-hour-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1 --state-rep full
wait
python3 main.py --algo ppo --penetration-rate 0.1 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full
