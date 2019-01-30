#!/bin/bash

## 2 maps for original / acktr 
## approx 7h30min long

cd ..

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo acktr --num-env-steps 10000000 --num-steps 32 --num-processes 16 --penetration-rate $i --env-name TrafficLight-Lust12408-rush-hour-v0 --state-rep original
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo acktr --num-env-steps 10000000 --num-steps 32 --num-processes 16 --penetration-rate $i --env-name TrafficLight-Lust12408-regular-time-v0 --state-rep original
	wait
done
