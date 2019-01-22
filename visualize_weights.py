import torch
import argparse
import os
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def main():

	args = get_args()

	actor_critic = torch.load(os.path.join(args.save_path,"model.pt"))

	print(actor_critic)

	fig = plt.figure()
	fig, axes = plt.subplots(nrows=4, ncols=2)

	i=0
	for m in actor_critic.modules():
		if isinstance(m,nn.Conv1d):
			kernels = m.weight.data.numpy()
			kernels = np.squeeze(kernels)
			im = axes.flat[i].imshow(kernels,cmap='gray')
			print(m.weight.data.numpy())
			i+=1
	fig.colorbar(im, ax=axes.ravel().tolist())
	plt.savefig(os.path.join(args.save_path,"weights.png"))



def get_args():

    parser = argparse.ArgumentParser(description='visualize_RL')
    parser.add_argument('--save-path', default='',
                        help='path of the saved model to evaluate (default: "")')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()