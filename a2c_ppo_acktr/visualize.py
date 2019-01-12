import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 8})


def visualize(rewards, algo):

    num_updates = len(rewards)
    fig = plt.figure()

    tx = np.arange(num_updates)    
    plt.plot(tx, rewards, label="{}".format(algo))

    plt.xlim(0,  num_updates+1)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Rewards')

    plt.title("TrafficLight-v0")
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    plt.savefig("./reward_results/test_fig.png")

if __name__ == "__main__":
    rewards = np.load("results.npy")
    visualize(rewards,"a2c")
