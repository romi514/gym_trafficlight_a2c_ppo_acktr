#!/bin/bash

cd ..

python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1 
--load-path /home/a2c/trained_models/acktr/TrafficLight-Lust12408-regular-time-v0/1/2019-01-31_16.40.27
wait

python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 30000000 --penetration-rate 1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 20000000 --penetration-rate 0.1 --load-path "$(echo /home/a2c/trained_models/a2c/TrafficLight-Lust12408-regular-time-v0/1/2*)"

