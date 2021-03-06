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



def test_on_highway_env():
    import gym
    import highway_env
    from DuelingDQN import DeepDuelingQNetwork
    from highway_car_dqn import DeepQnetwork
    env = gym.make('merge-v0')
    env = gym.wrappers.FlattenObservation(env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepQnetwork(obs_dim, action_dim)
    agent.QNet.load_and_save_weight('BreakoutDQN.weight', mode='load')
    record_video(env, agent.QNet)


def record_video(env, agent):
    import gym
    from gym.wrappers import Monitor
    import torch
    import highway_env
    env = Monitor(env, './video', force=True)
    obs = env.reset()
    eps = 10
    index = 0
    while index < eps:
        action = agent(torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda')))
        state_, reward, done, _ = env.step(action)
        if done:
            index += 1
    env.close()


if __name__ == '__main__':
    test_on_highway_env()