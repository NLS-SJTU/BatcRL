from utils import plot_learning_curve, record_video
from PPO import PPOAgent
from DDPG import DDPGAgent
from SAC import SACAgent
import gym
import time
from copy import  deepcopy
import torch
def demo_test():
    env_id = 'Pendulum-v0'
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim)
    agent_name = agent.__class__.__name__
    # using random explore to collect samples.
    total_step = 1e7
    eval_env = deepcopy(env)
    step = 0
    target_return = env.spec.reward_threshold
    if target_return is None:
        target_return = -200
    avg_return = -10**5
    t = time.time()
    step_record = []
    episode_return_mean = []
    episode_return_std = []
    print(f'env:{env_id}, obs dim:{obs_dim}, action dim:{action_dim}, target_return:{target_return}')
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
    # plot_learning_curve(step_record, episode_return_mean, episode_return_std, f'{env_id}_{agent_name}.png')
    # agent.evaluate(eval_env, render=True)
    record_video(agent, env, env_id, device=torch.device('cuda'))




if __name__ == '__main__':
    demo_test()