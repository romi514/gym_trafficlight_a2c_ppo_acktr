#!/bin/bash

cd ..

#Env adapt

python3 main.py --env-name TrafficLight-Lust12408-regular-time-eval-v0 --algo acktr --num-processes 1 --num-steps 1024 --penetration-rate 0.5 --state-rep original --load-path /home/a2c/trained_models/acktr/TrafficLight-Lust12408-regular-time-v0/0.5/2019-02-01_21.38.15/ --lr 7e-5
wait

python3 main.py --env-name TrafficLight-Lust12408-regular-time-eval-v0 --algo acktr --num-processes 1 --num-steps 1024 --penetration-rate 0.1 --state-rep original --load-path /home/a2c/trained_models/acktr/TrafficLight-Lust12408-regular-time-v0/0.1/2019-02-02_01.43.04/ --lr 7e-5
wait

python3 main.py --algo ppo --penetration-rate 0.5 --env-name TrafficLight-Lust12408-regular-time-eval-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/TrafficLight-Lust12408-regular-time-v0/0.5/2019-02-01_07.00.30
wait

python3 main.py --algo ppo --penetration-rate 0.1 --env-name TrafficLight-Lust12408-regular-time-eval-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/TrafficLight-Lust12408-regular-time-v0/0.1/2019-02-01_02.03.44/
wait

python3 main.py --algo ppo --penetration-rate 0.1 --env-name TrafficLight-Lust12408-rush-hour-eval-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/all_others/TrafficLight-Lust12408-rush-hour-v0/0.1/2019-01-31_17.09.08/
wait

python3 main.py --algo ppo --penetration-rate 0.5 --env-name TrafficLight-Lust12408-rush-hour-eval-v0 --lr 2.5e-4 --num-processes 1 --num-steps 1024 --num-mini-batch 4 --use-linear-clip-decay --state-rep original --load-path /home/a2c/trained_models/ppo/all_others/TrafficLight-Lust12408-rush-hour-v0/0.5/2019-02-01_10.27.10/
wait


