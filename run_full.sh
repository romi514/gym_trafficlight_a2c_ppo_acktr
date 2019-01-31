#!/bin/bash

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 # 9 * 10min = 180min = 1h30
do
	python3 main.py --algo ppo --state-rep full --penetration-rate $i --env-name TrafficLight-simple-sparse-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --num-env-steps 8000000
	wait
done

#declare -a envs=("TrafficLight-simple-dense-v0" "TrafficLight-simple-medium-v0" 
#	"TrafficLight-Lust12408-midnight-v0" "TrafficLight-Lust12408-rush-hour-v0" "TrafficLight-Lust12408-regular-time-v0")
#
#for i in "${envs[@]}"
#do
#	python3 main.py
#   echo "$i"
#done

