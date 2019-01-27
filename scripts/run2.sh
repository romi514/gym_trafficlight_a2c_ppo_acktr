#!/bin/bash

## 2 maps for original / acktr 
## approx 7h30min long

cd ..

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo acktr --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-simple-dense-v0 --state-rep original
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo acktr --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-Lust12408-midnight-v0 --state-rep original
	wait
done
