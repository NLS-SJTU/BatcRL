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
    env = gym.make('roundabout-v0')
    env = gym.wrappers.FlattenObservation(env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepDuelingQNetwork(obs_dim, action_dim)
    agent.DuelingQNet.load_and_save_weight('HighwayDQN.weight', mode='load')
    agent.evaluate(env, True)


if __name__ == '__main__':
    test_on_highway_env()