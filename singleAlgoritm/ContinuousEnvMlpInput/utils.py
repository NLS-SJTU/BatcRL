import os

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import torch
import gym

class FlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenObs, self).__init__(env)
        w, h = env.observation_space.shape
        env.observation_space.shape = (w * h, )
    def observation(self, observation):
        return observation.ravel()


def plot_learning_curve(steps, avg_return, std_return, name='plot_learning_curve.jpg'):
    steps = np.array(steps)
    avg_return = np.array(avg_return)
    std_return = np.array(std_return)
    plt.plot(steps, avg_return, label='Episode Return')
    plt.fill_between(steps, avg_return-std_return, avg_return + std_return, alpha=0.3)
    plt.legend()
    plt.savefig(name)
    plt.close()


def record_video(agent, env, save_dir,device=torch.device('cpu')):
    epochs = 5
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    index = 0
    images = []
    for _ in range(epochs):
        obs = env.reset()
        done = False
        while not done:
            img = env.render(mode='rgb_array')
            images.append(img)
            index += 1
            action = agent.select_action(obs)
            obs_, reward, done, _ = env.step(action)
            obs = obs_
            if done:
                break
    imageio.mimsave(f'{save_dir}.gif', images, fps=30)




if __name__ == "__main__":
    import highway_env
    env =gym.make('highway-v0')
    env = FlattenObs(env)
    print(env.reset().shape)
    print(env.observation_space)