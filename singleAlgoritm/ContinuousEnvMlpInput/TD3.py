from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import os


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size=1e6, device=torch.device('cpu')):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device
        self.state_buf = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.other_buf = torch.empty((max_size, action_dim + 2), dtype=torch.float32, device=device)
        self.index = 0
        self.total_len = 0

    def append(self, state, other):
        # other: reward, done, action
        self.index = self.index % self.max_size
        self.state_buf[self.index] = torch.as_tensor(state, device=self.device)
        self.other_buf[self.index] = torch.as_tensor(other, device=self.device)
        self.index += 1
        self.total_len = min(max(self.index, self.total_len), self.max_size)

    def sample_batch(self, batch_size):
        batch_index = np.random.randint(0, self.total_len - 1, batch_size)
        return (
            self.state_buf[batch_index],  # s_t
            self.other_buf[batch_index, 2:],  # a_t
            self.other_buf[batch_index, 0],  # reward
            self.other_buf[batch_index, 1],  # done
            self.state_buf[batch_index + 1]  # s_{t+1}
        )


class ActorTD3(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=256):
        super(ActorTD3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state).tanh()

    def get_actions(self, state, noise_std=0.5):
        action = self.net(state).tanh()
        noise = (torch.rand_like(action) * noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=256):
        super(CriticTD3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        latent = self.net(state_action)
        return self.q1(latent)

    def get_q1_q2(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        latent = self.net(state_action)
        # twin q value.
        q1 = self.q1(latent)
        q2 = self.q2(latent)
        return q1, q2


class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 1e-4
        self.batch_size = 512
        self.noise_std = 0.2  # must < 1
        self.mid_dim = 256
        self.target_step = 4096
        self.repeat_time = 32
        self.gamma = 0.99
        self.max_memory = int(1e7)
        self.tau = 2 ** -8  # soft update
        self.update_freq = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ActorTD3(state_dim, action_dim, self.mid_dim).to(self.device)
        self.critic = CriticTD3(state_dim, action_dim, self.mid_dim).to(self.device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.buffer = ReplayBuffer(state_dim, action_dim, self.max_memory, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.last_state = None

    def explore_env(self, env) -> int:
        obs = env.reset() if self.last_state is None else self.last_state
        for _ in range(self.target_step):
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            other = (reward, (1 - done) * self.gamma, *action)
            self.buffer.append(obs, other)
            obs = env.reset() if done else next_obs
        self.last_state = obs
        return self.target_step

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.actor.get_actions(state, self.noise_std)
        return action.detach().cpu().numpy()[0]

    def update(self) -> None:
        for step in range(int(self.target_step * self.repeat_time / self.batch_size)):
            state, action, reward, mask, state_ = self.buffer.sample_batch(self.batch_size)
            with torch.no_grad():
                # different with ddpg, here, ddpg doesn't add noise to action, and ddpg doesn't have twin q function
                q_target = reward + mask * torch.min(*self.target_critic.get_q1_q2(state_, self.target_actor.get_actions(state_, self.noise_std))).squeeze()
            q1, q2 = self.critic.get_q1_q2(state, action)
            val_loss = 0.5 * self.mse_loss(q1.squeeze(), q_target) + 0.5 * self.mse_loss(q2.squeeze(), q_target)
            self.optim_update(self.critic_optimizer, val_loss)
            if step % self.update_freq == 0:
                actor_obj = -self.critic(state, self.actor(state)).mean()
                self.optim_update(self.actor_optimizer, actor_obj)
                self.soft_update(self.actor, self.target_actor, self.tau)
                self.soft_update(self.critic, self.target_critic, self.tau)

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

    @staticmethod
    def soft_update(q_eval, q_target, tau):
        for q_eval_params, q_target_params in zip(q_eval.parameters(), q_target.parameters()):
            q_target_params.data.copy_(tau * q_eval_params.data + (1 - tau) * q_target_params.data)

# demo
# Pendulum: 15 minutes