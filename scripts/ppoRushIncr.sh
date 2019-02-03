#!/bin/bash

cd ..

python3 main.py --algo ppo --penetration-rate 0.8 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/1/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.7 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.8/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.6 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.7/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.5 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.6/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.4 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.5/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.3 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.4/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.2 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.3/2*)" --eval-interval 400
wait

python3 main.py --algo ppo --penetration-rate 0.1 --env-name TrafficLight-Lust12408-rush-hour-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-Lust12408-rush-hour-v0/0.2/2*)" --eval-interval 400



