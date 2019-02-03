#!/bin/bash

cd ..

python3 main.py --algo ppo --penetration-rate 0.5 --env-name TrafficLight-simple-dense-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-simple-dense-v0/0.4/2*)" --eval-interval 400 --num-env-steps 8000000
wait

python3 main.py --algo ppo --penetration-rate 0.6 --env-name TrafficLight-simple-dense-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-simple-dense-v0/0.5/2*)" --eval-interval 400 --num-env-steps 8000000
wait

python3 main.py --algo ppo --penetration-rate 0.7 --env-name TrafficLight-simple-dense-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-simple-dense-v0/0.6/2*)" --eval-interval 400 --num-env-steps 8000000
wait

python3 main.py --algo ppo --penetration-rate 0.8 --env-name TrafficLight-simple-dense-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-simple-dense-v0/0.7/2*)" --eval-interval 400 --num-env-steps 8000000
wait

python3 main.py --algo ppo --penetration-rate 1 --env-name TrafficLight-simple-dense-v0 --lr 2.5e-4 --num-processes 8 --num-steps 128 --num-mini-batch 4 --use-linear-lr-decay --use-linear-clip-decay --state-rep original --load-path "$(echo /home/a2c/trained_models/ppo/TrafficLight-simple-dense-v0/0.8/2*)" --eval-interval 400 --num-env-steps 8000000


