#!/bin/bash

# Last maps for sign representation for all algos
# approx 7h30min long

cd ..


python3 main.py --algo a2c --num-env-steps 2000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 16
wait
python3 main.py --algo acktr --num-env-steps 2000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 16
wait

python3 main.py --algo a2c --num-env-steps 4000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 32
wait
python3 main.py --algo acktr --num-env-steps 4000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 32
wait

python3 main.py --algo a2c --num-env-steps 8000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 64
wait
python3 main.py --algo acktr --num-env-steps 8000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 64
wait

python3 main.py --algo a2c --num-env-steps 12000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 128
wait
python3 main.py --algo acktr --num-env-steps 12000000 --penetration-rate 0.4 --env-name TrafficLight-simple-medium-v0 --state-rep full --num-steps 128
wait