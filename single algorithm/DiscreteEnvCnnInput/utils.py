import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(steps, avg_return, std_return, name='plot_learning_curve.jpg'):
    steps = np.array(steps)
    avg_return = np.array(avg_return)
    std_return = np.array(std_return)
    plt.plot(steps, avg_return, label='Episode Return')
    plt.fill_between(steps, avg_return-std_return, avg_return + std_return, alpha=0.3)
    plt.legend()
    plt.savefig(name)