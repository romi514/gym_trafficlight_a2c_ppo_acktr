#!/bin/bash

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1
do
	python3 main.py --num-env-steps 5000000 --penetration-rate $i --env-name TrafficLight-simple-sparse-v0
	wait
done