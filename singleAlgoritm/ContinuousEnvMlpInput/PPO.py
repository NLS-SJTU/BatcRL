import torch
import torch.nn as nn
import numpy as np
import os


class ReplayBuffer:
    def __init__(self, state_dim: int,action_dim:int, max_size: int = 10000, device=torch.device('cpu')):
        self.max_size = max_size
        self.device = device
        self.action_dim = action_dim
        self.state_buffer = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        # r_t, done,a_t, noise
        self.other_buffer = torch.empty((max_size, 2+ 2*action_dim), dtype=torch.float32, device=device)
        self.index = 0

    def append(self, state, other):
        self.index = self.index % self.max_size
        self.state_buffer[self.index] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.other_buffer[self.index] = torch.as_tensor(other, dtype=torch.float32, device=self.device)
        self.index += 1

    def sample_all(self, device):
        return (
            torch.as_tensor(self.state_buffer[:self.index], device=device),  # s_t
            torch.as_tensor(self.other_buffer[:self.index, 0], dtype=torch.long, device=device),  # r_t
            torch.as_tensor(self.other_buffer[:self.index, 1], device=device),  # done
            torch.as_tensor(self.other_buffer[:self.index, 2: 2+self.action_dim], device=device),  # a_t
            torch.as_tensor(self.other_buffer[:self.index, -self.action_dim:], device=device),  # noise
        )

    def empty_buffer(self):
        self.index = 0


class ActorPPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mid_dim=512):
        super(ActorPPO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
            nn.Linear(mid_dim, action_dim)
        )
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, states):
        return self.net(states)

    def get_action(self, states):
        action_avg = self.net(states)
        action_std = self.action_std_log.exp()
        noises = torch.rand_like(action_avg)
        actions = action_avg + noises * action_std  # N(avg, std)
        return actions, noises

    def get_action_logprob_entropy(self, states, actions):
        action_avg = self.net(states)
        action_std = self.action_std_log.exp()
        '''
        pi(a|s)=N(u, delta)
        log pi(a|s)=log(N(u, delta))= -log(delta) - log(\sqrt(2pi) - 0.5*((x-u)/delta)**2)
        entropy = \sum [p(x) * log(p(x))]
        '''
        delta = 0.5 * ((actions - action_avg) / action_std).pow(2)
        logprob = -(delta + action_std + self.log_sqrt_2pi).sum(1)
        entropy = (logprob * logprob.exp()).mean()
        return logprob, entropy

    def get_old_logprob(self, noise):
        '''
        :param noise: sample from N(0, 1), i.e: it has included (x-u)/delta, convert a random gaussian dist to N(0, 1)
        :return: under such condition, the logprob can be expressed by the following equation.
        '''
        delta = noise.pow(2) * 0.5
        return -(self.action_std_log + self.log_sqrt_2pi + delta).sum(1)

class CriticPPO(nn.Module):
    def __init__(self, state_dim: int, mid_dim: int = 512):
        super(CriticPPO, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim,mid_dim),nn.Hardswish(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, state):
        return self.state_value(state)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = 256
        self.actor_lr = 2e-4
        self.critic_lr = 1e-4
        self.entropy_coef = 0.002
        self.batch_size = 512
        self.clip_epsilon = 0.2
        self.target_step = 2048
        self.max_buffer_size = self.target_step * 10
        self.repeat_time = 1
        self.reward_scale = 1.
        self.tau = 2 ** -8  # soft update.
        self.gamma = 0.98  # discount factor.
        self.explore_rate = 0.75

        self.net_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = ReplayBuffer(state_dim=self.state_dim,action_dim=self.action_dim,max_size=self.max_buffer_size, device=self.buffer_device)

        self.actor = ActorPPO(self.state_dim, self.action_dim, self.mid_dim).to(self.net_device)
        self.critic = CriticPPO(self.state_dim, self.mid_dim).to(self.net_device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.last_state = None
        self.mse_loss = nn.SmoothL1Loss()

    def select_action(self, state):
        state = torch.as_tensor((state, ), dtype=torch.float32, device=self.net_device)
        if np.random.rand() < self.explore_rate:
            actions, noises = self.actor.get_action(state)
        else:
            actions = self.actor(state)
            noises = torch.rand_like(actions)
        return actions.detach().cpu().numpy(), noises.detach().cpu().numpy()

    def explore_env(self, env):
        state = env.reset() if self.last_state is None else self.last_state
        for i in range(self.target_step):
            action, noise =[arr[0] for arr in self.select_action(state)]
            state_, reward, done, _ = env.step(np.tanh(action))
            others = (reward * self.reward_scale, (1 - done) * self.gamma, *action, *noise)
            self.replay_buffer.append(state, others)
            state = env.reset() if done else state_
        self.last_state = state
        return self.target_step

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    def update(self):
        with torch.no_grad():
            states, reward, mask, actions, noise = self.replay_buffer.sample_all(self.net_device)
            state_values = self.critic(states)
            old_logprob = self.actor.get_old_logprob(noise)

            discount_cumulative_rewards = torch.empty(len(actions), dtype=torch.float32, device=self.net_device)
            tmp_last_state = torch.as_tensor((self.last_state,), dtype=torch.float32, device=self.net_device)
            last_value = self.critic(tmp_last_state)
            for i in range(len(actions) - 1, -1, -1):
                discount_cumulative_rewards[i] = reward[i] + mask[i] * last_value
                last_value = discount_cumulative_rewards[i]

            advantage = discount_cumulative_rewards - (mask * state_values.squeeze(1))
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        for _ in range(int(self.target_step * self.repeat_time / self.batch_size)):
            indices = np.random.randint(0, len(actions), self.batch_size)
            batch_state = states[indices]
            batch_action = actions[indices]
            batch_dis_cum_rewards = discount_cumulative_rewards[indices]
            batch_advantage = advantage[indices]
            batch_old_logprob = old_logprob[indices]
            batch_new_logprob, batch_entropy = self.actor.get_action_logprob_entropy(batch_state, batch_action)
            ratio = (batch_new_logprob - batch_old_logprob.detach()).exp()
            surrogate1 = batch_advantage * ratio
            surrogate2 = batch_advantage * ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            actor_loss = obj_surrogate + batch_entropy * self.entropy_coef
            self.optim_update(self.actor_optimizer, actor_loss)
            batch_values = self.critic(batch_state).squeeze(1)
            critic_loss = self.mse_loss(batch_values, batch_dis_cum_rewards) / (batch_dis_cum_rewards.std() + 1e-6)
            self.optim_update(self.critic_optimizer, critic_loss)
        self.replay_buffer.empty_buffer()

    @torch.no_grad()
    def evaluate(self, env, render=False):
        epochs = 20
        res = np.zeros((epochs,))
        obs = env.reset()
        index = 0
        while index < epochs:
            if render: env.render()
            obs =torch.as_tensor((obs,), dtype=torch.float32, device=self.net_device)
            action, _ = self.actor.get_action(obs)
            action = action.detach().cpu().numpy()[0]
            s_, reward, done, _ = env.step(np.tanh(action))
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
