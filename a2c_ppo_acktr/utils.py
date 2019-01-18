import torch
import torch.nn as nn
import time
import datetime

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2: # if x is a batch of 1 dim vectors (sign)
            bias = self._bias.t().view(1, -1)
        else: # else we are in full rep
            bias = self._bias.t().view(1, -1, 1)

        return x + bias

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_time():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H.%M.%S')

def save_params(args, file):
    f = open(file, 'a')
    f.write('#################################\n')
    for key in vars(args):
        value = getattr(args, key)
        if value is None:
            value = "None"
        if isinstance(value,bool):
            value = str(value)
        f.write('{:<20}\t{:<20}\n'.format(key,value))
    f.write('#################################\n')
    f.close()






