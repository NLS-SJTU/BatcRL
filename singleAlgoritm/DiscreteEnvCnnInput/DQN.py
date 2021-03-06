'''
implemented by PyTorch.
'''
import gym
import numpy as np
import torch.nn as nn
import torch
from torch.optim import RMSprop
from typing import Tuple
import os


class ReplayBuffer:
    def __init__(self, state_dim:list, max_size=10000, device=torch.device('cpu')):
        self.device = device
        self.state_buffer = torch.empty((max_size, *state_dim), dtype=torch.float32, device=device)
        self.other_buffer = torch.empty((max_size, 3), dtype=torch.float32, device=device)
        self.index = 0
        self.max_size = max_size
        self.total_len = 0

    def append(self, state, other):
        self.index = self.index % self.max_size
        self.total_len = max(self.index, self.total_len)
        self.state_buffer[self.index] = torch.as_tensor(state, device=self.device)
        self.other_buffer[self.index] = torch.as_tensor(other, device=self.device)
        self.index += 1

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, self.total_len - 1, batch_size)
        return (
            self.state_buffer[indices],  # S_t
            self.other_buffer[indices, 2:].long(),  # a_t
            self.other_buffer[indices, 0],  # r_t
            self.other_buffer[indices, 1],  # done
            self.state_buffer[indices + 1]
        )


class QNet(nn.Module):
    def __init__(self, img_dim: list, action_dim: int, mid_dim: int = 512) -> None:
        '''
        :param img_dim: (size, size, channel). e.g: 28 * 28 * 3
        :param action_dim: the number of actions.
        :param mid_dim: mlp dim.
        '''
        super(QNet, self).__init__()
        channel, size, _ = img_dim
        self.action_dim = action_dim
        cnn = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            tmp = torch.rand((1, channel, size, size))
            tmp_dim = cnn(tmp).shape[1]
        mlp = nn.Sequential(
            nn.Linear(tmp_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )
        self.encoder = nn.Sequential(cnn, mlp)

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        # return Q(s, a). the estimated state-action value.
        return self.encoder(state)

    def load_and_save_weight(self, path, mode='load'):
        if mode == 'load':
            if os.path.exists(path):
                self.load_state_dict(torch.load(path))
        else:
            torch.save(self.state_dict(), path)


class DQNAgent:
    def __init__(self, obs_dim: list, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_tate = 0.00025
        self.tau = 2 ** -8  # soft update.
        self.gamma = 0.99  # discount factor.
        self.batch_size = 128
        self.memory_size = 100000
        self.explore_rate = 0.2  # epsilon greedy rate.
        '''
        for exploring in the env, each time will collect self.target_step * self.batch_size number of samples into buffer,
        for updating neural network, each time will update self.target_step * self.repeat_time times. 
        '''
        self.target_step = 1024
        self.repeat_time = 2
        self.reward_scale = 1.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayBuffer(obs_dim, self.memory_size, self.device)
        self.QNet = QNet(obs_dim, action_dim).to(self.device)
        self.QNet_target = QNet(obs_dim, action_dim).to(self.device)  # Q target.
        self.optimizer = RMSprop(self.QNet.parameters(), self.learning_tate,alpha=0.95, eps=0.01)
        self.loss_func = nn.MSELoss(reduction='mean')

    def select_action(self, state: np.ndarray) -> int:
        # using epsilon greedy algorithm to select the action.
        if np.random.random() < self.explore_rate:  # epsilon greedy.
            action = np.random.randint(self.action_dim)
        else:
            state = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            dist = self.QNet(state)[0]
            action = dist.argmax(dim=0).cpu().numpy()
        return action

    def explore_env(self, env, all_greedy=False) -> int:
        # to collect samples into replay buffer.
        state = env.reset()
        for _ in range(self.target_step):
            action = np.random.randint(self.action_dim) if all_greedy else self.select_action(state)
            state_, reward, done, _ = env.step(action)
            other = (reward * self.reward_scale, 0.0 if done else self.gamma, action)
            self.buffer.append(state, other)
            state = env.reset() if done else state_
        return self.target_step

    @staticmethod
    def soft_update(eval_net, target_net, tau) -> None:
        # soft update for network. the equation: W_1 * tau + W_2 * (1 - tau)
        for target_param, local_param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update(self) -> None:
        # update the neural network.
        for _ in range(int(self.target_step * self.repeat_time / self.batch_size)):
            state, action, reward, mask, state_ = self.buffer.sample_batch(self.batch_size)
            # Q(s_t, a_t) = r_t + \gamma * max Q(s_{t+1}, a)
            next_q = self.QNet_target(state_).detach().max(1)[0]
            q_target = reward + mask * next_q
            q_eval = self.QNet(state).gather(1, action)
            loss = self.loss_func(q_eval, q_target.view(self.batch_size, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(self.QNet, self.QNet_target, self.tau)

    def evaluate(self, env, render=False):
        epochs = 20
        res = np.zeros((epochs,))
        obs = env.reset()
        index = 0
        while index < epochs:
            if render: env.render()
            obs = torch.as_tensor((obs,), dtype=torch.float32, device=self.device).detach_()
            dist = self.QNet(obs)[0]
            action = int(dist.argmax(dim=0).cpu().numpy())
            print(action)
            s_, reward, done, _ = env.step(action)
            res[index] += reward
            if done:
                print('done')
                index += 1
                obs = env.reset()
            else:
                obs = s_
        return res.mean(), res.std()





def demo_test():
    import time
    from copy import deepcopy
    from AtrariEnv import make_env
    torch.manual_seed(100)
    env_id = '' # 'CartPole-v0'
    env = make_env(env_id)
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim)
    agent_name = agent.__class__.__name__

    # using random explore to collect samples.
    agent.explore_env(deepcopy(env), all_greedy=True)
    total_step = 10000000
    eval_env = deepcopy(env)
    step = 0
    target_return = 350
    avg_return = 0
    t = time.time()
    step_record = []
    episode_return_mean = []
    episode_return_std = []
    init_save = 100000
    from utils import plot_learning_curve
    while step < total_step and avg_return < target_return - 1:
        step += agent.explore_env(env)
        agent.update()
        avg_return, std_return = agent.evaluate(eval_env)
        print(f'current step:{step}, episode return:{avg_return}')
        episode_return_mean.append(avg_return)
        episode_return_std.append(std_return)
        step_record.append(step)
        plot_learning_curve(step_record, episode_return_mean, episode_return_std, f'{env_id}_{agent_name}_plot_learning_curve.jpg')
        if step > init_save:
            agent.QNet.load_and_save_weight(f'{env_id}_{agent_name}.weight', mode='save')
            init_save += init_save

    agent.QNet.load_and_save_weight(f'{env_id}_{agent_name}.weight', mode='load')
    t = time.time() - t
    print('total cost time:', t, 's')
    plot_learning_curve(step_record, episode_return_mean, episode_return_std,f'{env_id}_{agent_name}_plot_learning_curve.jpg')
    agent.evaluate(env, render=True)


if __name__ == '__main__':
    demo_test()
