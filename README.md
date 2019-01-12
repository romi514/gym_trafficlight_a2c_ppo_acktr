# gym_trafficlight_a2c_ppo_acktr

## Introduction

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)


This implementation is a modified version of the public repo of Kostrikov and Ilya - [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

It has been adapted to work with the gym environemnt [gym_trafficlight](https://github.com/beedrill/gym_trafficlight)

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom) (not used yet, but may be used in the future)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
pip3 install torch torchvision

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Gym TrafficLight-v0 environment
git clone https://github.com/beedrill/gym_trafficlight.git
cd gym_trafficlight
pip install -e .

# Other requirements
pip install -r requirements.txt
```
### SUMO
see SUMO installation [here]https://github.com/beedrill/gym_trafficlight

## Docker

Install docker [here](https://www.docker.com/) and run the following commands :


### CPU
```bash
docker run -it --name rltl_baselines -e SUMO_HOME='/home/sumo' -e OPENAI_LOGDIR='/home/training_logs' -e OPENAI_LOG_FORMAT='stdout,csv,tensorboard' -v /path/to/package/gym_trafficlight:/home/gym_trafficlight -v /path/to/package/baselines:/home/baselines -v /path/to/package/gym_trafficlight_a2c_ppo_acktr:/home/gym_trafficlight_a2c_ppo_acktr  beedrill/rltl-docker:cpu-py3 /bin/bash
```
### GPU (not tested) - with torch.cuda

```bash
docker run -it --name rltl_baselines_gpu -e SUMO_HOME='/home/sumo' -e OPENAI_LOGDIR='/home/training_logs' -e OPENAI_LOG_FORMAT='stdout,csv,tensorboard' -v /path/to/package/gym_trafficlight:/home/gym_trafficlight -v /path/to/package/baselines:/home/baselines -v /path/to/package/gym_trafficlight_a2c_ppo_acktr:/home/a2c  beedrill/rltl-docker:gpu-py3 /bin/bash
```

## Training

To train with default arguments (see `main.py --help ` to see or `a2c_ppo_acktr/arguments.py`)
 
```bash
python3 main.py
```
A2C is the algorithm running by default

## Evaluation

To evaluate a saved model with sumo-gui visualization. Can precise number of steps with --num_steps

```bash
python3 evaluate_model.py --model_path ./trained_models/a2c/TrafficLight-v0-best.pt
```





