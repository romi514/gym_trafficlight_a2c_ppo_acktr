import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.rcParams.update({'font.size': 8})


def visualize(rewards, algo, save_path, n=400):

    num_updates = len(rewards)
    if n > num_updates:
        n = num_updates

    if num_updates%n != 0:
        rewards = rewards[:-(num_updates%n)]

    sectioned_rew = np.reshape(rewards,(n,int(num_updates/n)))
    means = np.mean(sectioned_rew,axis=1)

    fig = plt.figure()

    plt.plot(np.arange(n), means,label="means")

    plt.xlim(0,  n+1)

    plt.xlabel('Number of updates (x{})'.format(int(num_updates/n)))
    plt.ylabel("Average Rewards over {} updates".format(int(num_updates/n)))

    plt.title("TrafficLight-v0")
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    plt.savefig(os.path.join(save_path,'average_rewards.png'))

if __name__ == "__main__":
    rewards = np.load("results.npy")
    visualize(rewards,"a2c")
