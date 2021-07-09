'''
implemented by PyTorch.
'''
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
import gym

class ReplayBuffer:
    def __init__(self, obs_dim: int,batch_size:128,  memory_size: int = 2 * 10) -> None:
        self._memory_size = memory_size
        self.obs_dim = obs_dim
        self.buffer = np.zeros((self._memory_size, obs_dim * 2 + 3))
        self._batch_size = batch_size
        self._memory_counter = 0

    def store_transition(self,s: np.ndarray, a: int, r: float, done:float, s_:np.ndarray) -> None:
        transition = np.hstack((s, [a, r, done], s_))
        index = self._memory_counter % self._memory_size
        self.buffer[index, :] = transition
        self._memory_counter += 1

    def sample_batch(self):
        index = np.random.choice(self._memory_size, self._batch_size)
        batch_state = torch.FloatTensor(self.buffer[index, :self.obs_dim])
        batch_action = torch.LongTensor(self.buffer[index, self.obs_dim:self.obs_dim+1].astype(int))
        batch_reward = torch.FloatTensor(self.buffer[index, self.obs_dim+1:self.obs_dim+2])
        batch_mask = torch.FloatTensor(self.buffer[index, self.obs_dim+2:self.obs_dim+3])
        batch_state_ = torch.FloatTensor(self.buffer[index, -self.obs_dim:])
        return batch_state, batch_action, batch_reward, batch_mask, batch_state_


class QNet(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int, mid_dim:int=256) -> None:
        super(QNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        return self.encoder(state)



class DeepQnetwork:
    def __init__(self, obs_dim:int, action_dim:int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learn_rate = 1e-4
        self.tau = 0.9 # soft update.
        self.gamma = 0.99 # discount factor.
        self.batch_size = 128
        self.memory_size = 4096
        self.explore_rate = 0.2 # epsilon greedy rate.
        self.target_step = 1024
        self.repect_time = 10
        self.reward_scale = 1.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayBuffer(obs_dim, action_dim, self.memory_size)
        self.QNet = QNet(obs_dim, action_dim).to(self.device)
        self.QNet_target = QNet(obs_dim, action_dim).to(self.device)
        self.optimizer = Adam(self.QNet.parameters(), self.learn_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state:np.ndarray) -> int:
        if np.random.random() < self.explore_rate: # epsilon greedy.
            action = np.random.randint(self.action_dim)
        else:
            state = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            dist = self.QNet(state)[0]
            action = dist.argmax(dim=0).cpu().numpy()
        return action

    def explore_env(self, env) -> None:
        state = env.reset()
        for _ in range(self.target_step):
            action = self.select_action(state)
            state_, reward, done, _ = env.step(action)
            self.buffer.store_transition(state, action, reward * self.reward_scale, 0. if done else self.gamma, state_)
            state = env.reset() if done else state_

    def soft_update(self) -> None:
        """soft update a target network via current network
        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(self.QNet_target.parameters(), self.QNet.parameters()):
            tar.data.copy_(cur.data.__mul__(self.tau) + tar.data.__mul__(1 - self.tau))

    def update(self) -> None:
        for _ in range(self.target_step * self.repect_time):
            with torch.no_grad():
                state, action, reward, mask, state_ = self.buffer.sample_batch()
                next_q = self.QNet_target(state_).max(dim=1, keepdim=True)[0]
                q_taget = reward + mask * next_q
            q_eval = self.QNet(state).gather(1, action)
            self.optimizer.zero_grad()
            loss = self.criterion(q_eval, q_taget)
            loss.backward()
            self.optimizer.step()
            self.soft_update()






def demo_test():
    def test_agent(env, agent, times=100):
        res = []
        for _ in range(times):
            r = 0.
            obs = env.reset()
            while True:
                action = agent.select_action(obs)
                obs_, reward, done, _ = env.step(action)
                r += reward
                if done:
                    res.append(r)
                    break
        return np.array(res).mean()







    import gym
    from copy import  deepcopy
    env = gym.make('CartPole-v0')
    obs_dim =env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepQnetwork(obs_dim,action_dim)
    epochs = 1000
    for epoch in range(epochs):
        agent.explore_env(deepcopy(env))
        agent.update()
        if epoch % 20 == 0:
            mr = test_agent(deepcopy(env), deepcopy(agent))
            print(f'current epoch:{epoch}, reward:{mr}')


if __name__ == '__main__':
    demo_test()










