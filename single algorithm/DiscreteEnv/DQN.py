'''
implemented by PyTorch.
'''
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from typing import Tuple


class ReplayBuffer:
    def __init__(self, obs_dim: int, batch_size: int = 128, memory_size: int = 2 * 10,
                 device=torch.device('cpu')) -> None:
        '''
        :param obs_dim: the dim of observation. type: int. for gym env: obs_dim = env.observation_space.shape[0]
        :param batch_size: each time the class will sample the "batch_size" items from ReplayBuffer.
        :param memory_size:  total size of ReplayBuffer.
        :param device: when using sample func, the samples will put into cpu or gpu.
        '''
        self._memory_size = memory_size
        self.obs_dim = obs_dim
        # column: (s, a, r, done, s_) = 2 * obs_dim + 3 constants.
        self.buffer = np.zeros((self._memory_size, obs_dim * 2 + 3), dtype=np.float32)
        self._batch_size = batch_size
        self._memory_counter = 0
        self.device = device

    def store_transition(self, s: np.ndarray, a: int, r: float, done: float, s_: np.ndarray) -> None:
        transition = np.hstack((s, a, r, done, s_))
        index = self._memory_counter % self._memory_size
        self.buffer[index, :] = transition
        self._memory_counter += 1

    def sample_batch(self):
        index = np.random.choice(min(self._memory_counter, self._memory_size), self._batch_size)
        batch_state = torch.as_tensor(self.buffer[index, :self.obs_dim], dtype=torch.float32, device=self.device)
        batch_action = torch.as_tensor(self.buffer[index, self.obs_dim:self.obs_dim + 1], dtype=torch.long,
                                       device=self.device)
        batch_reward = torch.as_tensor(self.buffer[index, self.obs_dim + 1:self.obs_dim + 2], dtype=torch.float32,
                                       device=self.device)
        batch_mask = torch.as_tensor(self.buffer[index, self.obs_dim + 2:self.obs_dim + 3], dtype=torch.float32,
                                     device=self.device)
        batch_state_ = torch.as_tensor(self.buffer[index, -self.obs_dim:], dtype=torch.float32, device=self.device)
        return batch_state, batch_action, batch_reward, batch_mask, batch_state_


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, mid_dim: int = 128) -> None:
        '''
        :param obs_dim:  the dim of observation. type: int. for gym env: obs_dim = env.observation_space.shape[0]
        :param action_dim: action space, i.e: The number of actions that can be taken at each step. type:int. for gym env: action_dim = env.action_space.n
        :param mid_dim: hidden size of MLP.
        '''
        super(QNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        # return Q(s, a). the estimated state-action value.
        return self.encoder(state)


class DeepQnetwork:
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_tate = 5e-4
        self.tau = 1e-3  # soft update.
        self.gamma = 0.99  # discount factor.
        self.batch_size = 1024
        self.memory_size = 10000
        self.explore_rate = 0.2  # epsilon greedy rate.
        '''
        for exploring in the env, each time will collect self.target_step * self.batch_size number of samples into buffer,
        for updating neural network, each time will update self.target_step * self.repeat_time times. 
        '''
        self.target_step = 10
        self.repeat_time = 100
        self.reward_scale = 1.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayBuffer(obs_dim, self.batch_size, self.memory_size, self.device)
        self.QNet = QNet(obs_dim, action_dim).to(self.device)
        self.QNet_target = QNet(obs_dim, action_dim).to(self.device) # Q target.
        self.optimizer = Adam(self.QNet.parameters(), self.learning_tate)

    def select_action(self, state: np.ndarray) -> int:
        # using epsilon greedy algorithm to select the action.
        if np.random.random() < self.explore_rate:  # epsilon greedy.
            action = np.random.randint(self.action_dim)
        else:
            state = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            dist = self.QNet(state)[0]
            action = dist.argmax(dim=0).cpu().numpy()
        return action

    def explore_env(self, env, all_greedy=False) -> None:
        # to collect samples into replay buffer.
        state = env.reset()
        for _ in range(self.target_step * self.batch_size):
            action = np.random.randint(self.action_dim) if all_greedy else self.select_action(state)
            state_, reward, done, _ = env.step(action)
            self.buffer.store_transition(state, action, reward * self.reward_scale, 0. if done else self.gamma, state_)
            state = env.reset() if done else state_

    def soft_update(self) -> None:
        # soft update for network. the equation: W_1 * tau + W_2 * (1 - tau)
        for target_param, local_param in zip(self.QNet_target.parameters(), self.QNet.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self) -> None:
        # update the neural network.
        for _ in range(self.target_step * self.repeat_time):
            state, action, reward, mask, state_ = self.buffer.sample_batch()
            # Q(s_t, a_t) = r_t + \gamma * max Q(s_{t+1}, a)
            next_q = self.QNet_target(state_).detach().max(dim=1, keepdim=True)[0]
            q_target = reward + mask * next_q
            q_eval = self.QNet(state).gather(1, action)
            loss = nn.MSELoss(reduction='mean')(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update()


def demo_test():
    def test_agent(env, agent, times=50):
        res = []
        for _ in range(times):
            r = 0.
            obs = env.reset()
            while True:
                action = agent(torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))).detach_()
                action = action.argmax(dim=0).cpu().numpy()
                obs_, reward, done, _ = env.step(action)
                r += reward
                if done:
                    res.append(r)
                    break
        return np.array(res).mean()

    import gym
    from copy import deepcopy
    torch.manual_seed(0)
    env = gym.make('LunarLander-v2')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepQnetwork(obs_dim, action_dim)
    # using random explore to collect samples.
    agent.explore_env(deepcopy(env), all_greedy=True)
    epochs = 10000
    eval_env = deepcopy(env)
    for epoch in range(epochs):
        agent.explore_env(env)
        agent.update()
        if epoch % 5 == 0:
            mr = test_agent(eval_env, agent.QNet)
            print(f'current epoch:{epoch}, reward:{mr}')


if __name__ == '__main__':
    demo_test()
