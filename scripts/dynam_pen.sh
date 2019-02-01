#!/bin/bash

cd ..

python3 main.py --env-name TrafficLight-simple-medium-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1 --state-rep full
wait
python3 main.py --env-name TrafficLight-Lust12408-regular-time-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1 --state-rep full
wait

python3 main.py --env-name TrafficLight-simple-medium-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 30000000 --penetration-rate 0.1 --state-rep full
wait
python3 main.py --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 30000000 --penetration-rate 0.1 --state-rep full
wait