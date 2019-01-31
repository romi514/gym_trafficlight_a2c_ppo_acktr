#!/bin/bash

cd ..

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1
do
	python3 main.py --algo ppo --penetration-rate $i --env-name TrafficLight-Lust12408-regular-time-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full
	wait
done


for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1
do
	python3 main.py --algo ppo --penetration-rate $i --env-name TrafficLight-simple-medium-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep full
	wait
done