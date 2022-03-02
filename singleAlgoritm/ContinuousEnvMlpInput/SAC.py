import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class ReplayBuffer:
    def __init__(self, state_dim, action_dim,max_size=1e6, device = torch.device('cpu')):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device
        self.state_buf = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.other_buf = torch.empty((max_size, action_dim + 2), dtype=torch.float32, device=device)
        self.index = 0
        self.total_len = 0

    def append(self, state, other):
        #other: reward, done, action
        self.index = self.index % self.max_size
        self.state_buf[self.index] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.other_buf[self.index]  = torch.as_tensor(other, dtype=torch.float32, device=self.device)
        self.index = self.index+1
        self.total_len = min(self.max_size, max(self.index, self.total_len))

    def sample_batch(self, batch_size):
        self.idx = np.random.randint(0, self.total_len-1, batch_size)
        state = self.state_buf[self.idx]
        action = self.other_buf[self.idx, 2:]
        reward = self.other_buf[self.idx, 0].unsqueeze(1)
        mask = self.other_buf[self.idx, 1].unsqueeze(1)
        next_state = self.state_buf[self.idx + 1]
        return state, action, reward, mask, next_state



class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=256):
        super(ActorSAC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )
        self.avg = nn.Linear(mid_dim, action_dim)
        self.log_std = nn.Linear(mid_dim, action_dim)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        x = self.net(state)
        return self.avg(x).tanh()

    def get_actions(self, state):
        x = self.net(state)
        avg = self.avg(x)
        std = self.log_std(x).clamp(-20, 2).exp()
        action = avg + torch.rand_like(avg) * std
        return action.tanh()

    def get_actions_logprob(self, state):
        x =self.net(state)
        avg = self.avg(x)
        log_std = self.log_std(x).clamp(-20, 2)
        noise = torch.rand_like(avg, requires_grad=True)
        action_tanh = (avg + noise * log_std.exp()).tanh()

        log_prob = log_std + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        log_prob = log_prob + (1. - action_tanh.pow(2)).log()
        # here ,log_prob lacks "minus sign"
        return action_tanh, log_prob.sum(1,keepdim=True)


class CriticSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=256):
        super(CriticSAC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU()
        )
        self.q1 = nn.Linear(mid_dim, 1) # optimal Q value
        self.q2 = nn.Linear(mid_dim, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = 0.1
        self.max_memo = int(1e6)
        self.target_step = 1024
        self.gamma = 0.99
        self.mid_dim = 256

        self.actor = ActorSAC(state_dim, action_dim, self.mid_dim).to(self.device)
        self.critic = CriticSAC(state_dim, action_dim,self.mid_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.max_memo, self.device)
        self.repect_time = 32
        self.state = None

    def select_action(self, states):
        states = torch.as_tensor((states, ), dtype=torch.float32, device=self.device)
        action = self.actor.get_actions(states)
        return action.cpu().detach().numpy()[0]

    def explore_env(self, env):
        state = self.state if self.state is not None else env.reset()
        for __ in range(self.target_step):
            action = self.select_action(state)
            state_, reward, done, _ = env.step(action)
            other = (reward, (1 - done) * self.gamma, *action)
            self.replay_buffer.append(state, other)
            state = state_ if not done else env.reset()
        self.state = state
        return self.target_step

    def update(self):
        for i in range(int(self.target_step * self.repect_time / self.batch_size)):
            batch_state, batch_action, batch_reward, batch_mask, batch_next_state = self.replay_buffer.sample_batch(self.batch_size)
            with torch.no_grad():
                next_action, next_log_prob = self.actor.get_actions_logprob(batch_next_state)
                next_q = torch.min(*self.critic(batch_next_state, next_action))
                q_label = batch_reward + batch_mask * (next_q + self.alpha * next_log_prob)
                # Q(s_t,a_t) = r_{t} + gamma * (Q(s_{t+1}, f(s_{t+1}) - logprob * alpha))
                # why we use "+", pls see get_actions_logprob func.

            # critic optim
            q1, q2 = self.critic(batch_state, batch_action)
            critic_loss = F.mse_loss(q1, q_label) + F.mse_loss(q2, q_label)
            self.optim_update(self.critic_optim, critic_loss)

            # actor optim
            action_pg, log_prob = self.actor.get_actions_logprob(batch_state)
            actor_obj = -torch.mean(torch.min(*self.critic(batch_state, action_pg)) + self.alpha * log_prob)
            self.optim_update(self.actor_optim, actor_obj)


    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @torch.no_grad()
    def evaluate(self, env, render=False):
        epochs = 20
        res = np.zeros((epochs,))
        obs = env.reset()
        index = 0
        while index < epochs:
            if render: env.render()
            obs = torch.as_tensor((obs,), dtype=torch.float32, device=self.device)
            action = self.actor(obs)
            action = action.detach().cpu().numpy()[0]
            s_, reward, done, _ = env.step(action)
            res[index] += reward
            if done:
                index += 1
                obs = env.reset()
            else:
                obs = s_
        return res.mean(), res.std()

    def load_and_save_weight(self, path, mode='load'):
        actor_path = os.path.join(path, 'actor.pth')
        critic_path = os.path.join(path, 'critic.pth')
        if mode == 'load':
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                self.actor.load_state_dict(torch.load(actor_path))
                self.critic.load_state_dict(torch.load(critic_path))
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)













