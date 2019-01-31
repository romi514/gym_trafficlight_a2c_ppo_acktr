#!/bin/bash

## Last maps for original / a2C 
## approx 6h40min long

cd ..

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo a2c --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-simple-dense-v0 --state-rep original
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo a2c --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-Lust12408-midnight-v0 --state-rep original
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo a2c --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-Lust12408-rush-hour-v0 --state-rep original
	wait
done

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1 
do
	python3 main.py --algo a2c --num-env-steps 1000000 --penetration-rate $i --env-name TrafficLight-Lust12408-regular-time-v0 --state-rep original
	wait
done