import torch.nn as nn
from torch.distributions import Categorical
import torch
import numpy as np


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=128):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, state):
        dist = Categorical(self.softmax(self.net(state)))
        action = dist.sample()
        return action

    def get_logprob(self, state, action):
        dist = Categorical(self.softmax(self.net(state)))
        logprob = dist.log_prob(action)
        return logprob

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.dones = []
        self.rewards = []

    def append(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample_all(self):
        return self.states, self.actions, self.rewards, self.dones

    def empty(self):
        self.states = []
        self.actions = []
        self.dones = []
        self.rewards = []



class PolicyGradient:
    def __init__(self, state_dim, action_dim):
        learning_rate = 2e-4
        self.actor = ActorNet(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.discount_factor = 0.98
        self.buffer = Buffer()
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.as_tensor((state,), dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(state)
        return action.detach().numpy()[0]

    def add_data(self,state, action, reward, done):
        self.buffer.append(state, action,reward, done)

    def update_network(self):
        states, actions, rewards, dones = self.buffer.sample_all()
        actor_loss_list = []
        # first calculate the reward to go value.\sum_{i=t}^\infty R_i
        reward_to_go = []
        tmp = 0
        for r in reversed(rewards):
            tmp = tmp * self.discount_factor + r
            reward_to_go = [tmp] + reward_to_go
        reward_to_go = torch.as_tensor(reward_to_go,dtype=torch.float32).unsqueeze(1)
        # calculate then advantage estimates, A_t(use td residual)
        actions = torch.as_tensor((actions,),dtype=torch.long)
        states = torch.as_tensor((states,), dtype=torch.float32)
        logprobs = self.actor.get_logprob(states, actions)
        self.policy_optimizer.zero_grad()
        policy_loss = -(logprobs * reward_to_go).mean()
        actor_loss_list.append(policy_loss.item())
        policy_loss.backward()
        self.policy_optimizer.step()
        self.buffer.empty()
        return np.array(actor_loss_list).mean()





def train_cartpole():
    import gym
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PolicyGradient(state_dim, action_dim)
    epochs = 6000
    return_list = []
    for i in range(epochs):
        obs = env.reset()
        episode_return = 0
        while True:
            action = agent.select_action(obs)
            obs_, reward, done, _ = env.step(action)
            episode_return +=reward
            agent.add_data(obs, action, reward, done)
            if done:
                actor_loss = agent.update_network()
                return_list.append(episode_return)
                if i %20 == 0:print(f'epochs:{i}, episode return:{episode_return}, actor loss:{actor_loss}')
                break
            obs = obs_
    plot_learning_curve(return_list)

import matplotlib.pyplot as plt
def plot_learning_curve(scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(range(len(running_avg)), running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
if __name__ == '__main__':
    train_cartpole()











