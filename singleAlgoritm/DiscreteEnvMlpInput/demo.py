from DQN import DQNAgent
from D3QN import D3QNAgent
from DoubleDQN import DoubleDQNAgent
from DuelingDQN import DuelingDQNAgent
from PPO import PPODiscreteAgent
from utils import plot_learning_curve, record_video
import gym
import time
from copy import  deepcopy
import torch
def demo_test():
    env_id = 'CartPole-v0' # 'CartPole-v0'
    algor = 'dqn'
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    algors = {
        'ppo':PPODiscreteAgent,
        'dqn':DQNAgent,
        'ddqn':DoubleDQNAgent,
        'duelingdqn':DuelingDQNAgent,
        'd3qn':D3QNAgent
    }
    agent = algors[algor.lower()](obs_dim, action_dim)
    agent_name = agent.__class__.__name__
    # using random explore to collect samples.
    total_step = 1e7
    eval_env = deepcopy(env)
    step = 0
    target_return = env.spec.reward_threshold
    if target_return is None:
        target_return = 10086
    avg_return = 0
    t = time.time()
    step_record = []
    episode_return_mean = []
    episode_return_std = []
    print(f'env:{env_id}, obs dim:{obs_dim}, action dim:{action_dim}, target_return:{target_return}')
    step += agent.explore_env(env)
    max_return = -10 ** 4
    while step < total_step and avg_return < target_return - 1:
        step += agent.explore_env(env)
        agent.update()
        avg_return, std_return = agent.evaluate(eval_env)
        if avg_return > max_return: max_return = avg_return
        print(f'current step:{step}, max return:{max_return}, episode return:{avg_return}')
        episode_return_mean.append(avg_return)
        episode_return_std.append(std_return)
        step_record.append(step)
        plot_learning_curve(step_record, episode_return_mean, episode_return_std,f'{env_id}_{agent_name}.png')
    agent.load_and_save_weight(f'{env_id}_{agent_name}', mode='load')
    t = time.time() - t
    print('total cost time:', t, 's')
    plot_learning_curve(step_record, episode_return_mean, episode_return_std, f'{env_id}_{agent_name}.png')
    agent.evaluate(eval_env, render=True)




if __name__ == '__main__':
    demo_test()