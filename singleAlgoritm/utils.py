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
    plt.close()


def record_video(env_id, agent):
    import gym
    from gym.wrappers import Monitor
    import torch
    import highway_env
    env = Monitor(gym.make(env_id), './video', force=True)
    obs = env.reset()
    eps = 10
    index = 0
    while index < eps:
        action = agent(torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda')))
        state_, reward, done, _ = env.step(action)
        if done:
            index += 1
    env.close()



