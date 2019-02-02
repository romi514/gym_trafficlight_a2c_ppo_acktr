#!/bin/bash

cd ..

python3 main.py --state-rep original --env-name TrafficLight-simple-sparse-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-simple-dense-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-midnight-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-rush-hour-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo acktr --num-processes 16 --num-steps 32 --use-linear-lr-decay --penetration-rate 0.1

#7h30
python3 main.py --state-rep original --env-name TrafficLight-simple-sparse-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-simple-dense-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-midnight-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-rush-hour-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 0.1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 0.1
wait

#7h30
python3 main.py --state-rep original --env-name TrafficLight-simple-sparse-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 1
wait
python3 main.py --state-rep original --env-name TrafficLight-simple-dense-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-midnight-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-rush-hour-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 1
wait
python3 main.py --state-rep original --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 10000000 --penetration-rate 1