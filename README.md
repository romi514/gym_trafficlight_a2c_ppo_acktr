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

## Installation

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
The [gym_trafficlight environment](https://github.com/beedrill/gym_trafficlight) is based on SUMO. If you are using docker, SUMO will be pre-installed in our docker image, otherwise, see SUMO installation [here](http://sumo.dlr.de/wiki/Installing)

### Install with Docker

Install docker [here](https://www.docker.com/) and run the following commands, __for GPU version, NVIDIA runtime is needed after installing docker, refer to [here](https://github.com/NVIDIA/nvidia-docker)__.


### CPU
```bash
docker run -it --name rltl_baselines -e SUMO_HOME='/home/sumo' -e OPENAI_LOGDIR='/home/training_logs' -e OPENAI_LOG_FORMAT='stdout,csv,tensorboard' -v /path/to/package/gym_trafficlight:/home/gym_trafficlight -v /path/to/package/baselines:/home/baselines -v /path/to/package/gym_trafficlight_a2c_ppo_acktr:/home/gym_trafficlight_a2c_ppo_acktr  beedrill/rltl-docker:cpu-py3 /bin/bash

cd /home/baselines
pip install -e .
cd /home/gym_trafficlight
pip install -e .
cd /home/gym_trafficlight_a2c_ppo_acktr
pip install -r requirement.txt

```
### GPU

```bash
nvidia-docker run -it --name rltl_pytorch_gpu -e SUMO_HOME='/home/sumo'   -v /path/to/package/gym_trafficlight:/home/gym_trafficlight  -v /path/to/package/gym_trafficlight_a2c_ppo_acktr:/home/a2c -v /path/to/package/baselines:/home/baselines  beedrill/rltl-docker:gpu-py3-pytorch /bin/bash

cd /home/baselines
pip install -e .
cd /home/gym_trafficlight
pip install -e .
cd /home/gym_trafficlight_a2c_ppo_acktr
pip install -r requirement.txt
```

To verify GPU device is using, open a python3 console and run:
```python
import torch
torch.cuda.current_device()
```
if it print out 0, you are good to go.
## Training

To train with default arguments (see `main.py --help ` to see or `a2c_ppo_acktr/arguments.py`)
<<<<<<< HEAD

=======
Use --vis to create average reward evolution plot
 
>>>>>>> master
```bash
python3 main.py
```
A2C is the algorithm running by default

## Evaluation

To evaluate a saved model with sumo-gui visualization. Can precise number of steps with --num_steps

```bash
python3 evaluate_model.py --model_path ./trained_models/a2c/TrafficLight-v0-best.pt
```
