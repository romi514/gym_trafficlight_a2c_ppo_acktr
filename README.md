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
* [OpenAI baselines](https://github.com/beedrill/baselines)
* [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) To visualize training progress

## Installation

In order to install requirements, follow:

```bash
# PyTorch
pip3 install torch torchvision

# Baselines for Atari preprocessing
git clone https://github.com/beedrill/baselines.git
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
docker run -it --name rltl_baselines -e SUMO_HOME='/home/sumo' -v /path/to/package/gym_trafficlight:/home/gym_trafficlight -v /path/to/package/baselines:/home/baselines -v /path/to/package/gym_trafficlight_a2c_ppo_acktr:/home/gym_trafficlight_a2c_ppo_acktr  beedrill/rltl-docker:cpu-py3 /bin/bash

pip3 install torch torchvision
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

pip3 install torch torchvision
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

To train with default arguments (see `main.py --help` to see or `a2c_ppo_acktr/arguments.py`)

### A2C (default)

```bash
python3 main.py
```

### PPO
```bash
python3 main.py --num-env-steps 1000000 --num-processes 4 --algo ppo 
```

###ACKTR
```bash
python3 main.py --num-env-steps 1000000 --algo acktr
```

Important arguments :
`--num-processes` - number of different environment running at the same time for training (default: 2)
`--penetration-rate` - Penetration rate for vehicle detection from 0 to 1 (default: 1)
`--penetration-type` - Penetration type over training, linear or constant (default: constant)
`--env-name` - Name of the map to train the network on (default: TrafficLight-v0)
`--num-env-steps` - Total number of environment steps combined (default: 10e7)
`--state-rep` - State representation, full, sign, or original (default: sign)
`--reward-type` - Reward type, partial, global, local (default: partial)

The number of updates is thus num-env-steps / num-processes / num-steps

By default, the path where the results, model, and parameters used are saved is `./trained_models/<algo>/<timestamp>/`

## Visualize

To visualize the training, use Tensorboard.
Open a new terminal outside of the docker, and navigate to `gym_trafficlight_a2c_ppo_acktr/trained_models/<algo>/<timestamp>/tb/`
Have tensorboard installed and type `tensorboard --logdir ./`, 
Copy the https address logged in the terminal window and navigate to it in your browser (on port 6006)

## Evaluation

To evaluate a saved model with sumo-gui visualization. Can precise number of steps with `--num_steps`

```bash
python3 evaluate_model.py --model_path ./trained_models/a2c/2019-01-20_19.35.56/model.pt
```
