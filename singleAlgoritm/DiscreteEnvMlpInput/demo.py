from DQN import DQNAgent
from D3QN import D3QNAgent
from utils import plot_learning_curve
import gym
import time
from copy import  deepcopy

def demo_test():
    env_id = 'CartPole-v0' # 'CartPole-v0'
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = D3QNAgent(obs_dim, action_dim)
    agent_name = agent.__class__.__name__
    # using random explore to collect samples.
    agent.explore_env(deepcopy(env), all_greedy=True)
    total_step = 1e7
    eval_env = deepcopy(env)
    step = 0
    target_return = 200
    avg_return = 0
    t = time.time()
    step_record = []
    episode_return_mean = []
    episode_return_std = []
    print(f'env:{env_id}, obs dim:{obs_dim}, action dim:{action_dim}:')

    while step < total_step and avg_return < target_return - 1:
        step += agent.explore_env(env)
        agent.update()
        avg_return, std_return = agent.evaluate(eval_env)
        print(f'current step:{step}, episode return:{avg_return}')
        episode_return_mean.append(avg_return)
        episode_return_std.append(std_return)
        step_record.append(step)
        plot_learning_curve(step_record, episode_return_mean, episode_return_std,f'{env_id}_{agent_name}.png')
    agent.load_and_save_weight(f'{env_id}_{agent_name}.weight', mode='save')
    t = time.time() - t
    print('total cost time:', t, 's')
    plot_learning_curve(step_record, episode_return_mean, episode_return_std, f'{env_id}_{agent_name}.png')
    agent.evaluate(eval_env, render=True)


if __name__ == '__main__':
    demo_test()