#!/bin/bash

# Last maps for sign representation for all algos
# approx 7h30min long

cd ..

for i in 16 32 64
do
	python3 main.py --algo a2c --num-env-steps 1000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps $i
	wait
	python3 main.py --algo acktr --num-env-steps 1000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps $i
	wait
done

for i in 128 256
do
	python3 main.py --algo a2c --num-env-steps 10000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps $i
	wait
	python3 main.py --algo acktr --num-env-steps 20000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps $i
	wait
done