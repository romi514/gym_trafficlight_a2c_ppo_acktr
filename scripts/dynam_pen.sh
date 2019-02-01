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


python3 main.py --env-name TrafficLight-Lust12408-regular-time-v0 --algo a2c --num-processes 10 --num-steps 16 --use-linear-lr-decay --num-env-steps 30000000 --penetration-rate 0.1 --state-rep full


python3 main.py --algo ppo --penetration-type linear --env-name TrafficLight-Lust12408-regular-time-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/TrafficLight-Lust12408-regular-time-v0/0.1/2019-02-01_02.03.44/
python3 main.py --algo ppo --penetration-type linear --env-name TrafficLight-simple-medium-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/TrafficLight-simple-medium-v0/0.1/01SimpleMed/

python3 main.py --algo a2c --penetration-type linear --env-name TrafficLight-simple-medium-v0 --num-processes 1 --num-steps 160 --use-linear-lr-decay --state-rep original --load-path /home/a2c/trained_models/a2c/TrafficLight-simple-medium-v0/0.1/2019-01-27_20.06.53/


python3 main.py --algo acktr --penetration-type linear --env-name TrafficLight-simple-medium-v0  --num-processes 1 --num-steps 512  --use-linear-lr-decay --state-rep original --load-path /home/a2c/trained_models/ppo/TrafficLight-simple-medium-v0/0.1/2019-02-01_02.03.44/

