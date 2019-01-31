#!/bin/bash

# Last maps for sign representation for all algos
# approx 7h30min long

cd ..

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 # 9 * 10min = 180min = 1h30
do
	python3 main.py --algo ppo --num-env-steps 700000 --penetration-rate $i --env-name TrafficLight-Lust12408-regular-time-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 # 9 * 10min = 180min = 1h30
do
	python3 main.py --algo ppo --num-env-steps 700000 --penetration-rate $i --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full
	wait
done