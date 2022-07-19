import torch
import torch.nn as nn
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, state_dim:int, max_size:int=10000, device=torch.device('cpu')):
        self.max_size = max_size
        self.device = device
        self.state_buffer = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        # r_t, done,a_t, action_log
        self.other_buffer = torch.empty((max_size, 4), dtype=torch.float32, device=device)
        self.index = 0

    def append(self, state, other):
        self.index = self.index % self.max_size
        self.state_buffer[self.index] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.other_buffer[self.index] = torch.as_tensor(other, dtype=torch.float32, device=self.device)
        self.index += 1

    def sample_all(self, device):
        return (
            torch.as_tensor(self.state_buffer[:self.index], device=device), # s_t
            torch.as_tensor(self.other_buffer[:self.index, 2], dtype=torch.long, device=device), # a_t
            torch.as_tensor(self.other_buffer[:self.index, 0], device=device), # r_t
            torch.as_tensor(self.other_buffer[:self.index, 1], device=device), # done
            torch.as_tensor(self.other_buffer[:self.index, 3],device=device), # log_prob
        )

    def empty_buffer(self):
        self.index = 0


class ActorPPO(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, mid_dim=512):
        super(ActorPPO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        return self.net(state)

    def get_action_and_logprob(self, state):
        prob = self.softmax(self.net(state))
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_logprob_entropy(self, state, action):
        prob = self.softmax(self.net(state))
        dist = torch.distributions.Categorical(prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return logprob, entropy


class CriticPPO(nn.Module):
    def __init__(self, state_dim:int, mid_dim:int=512):
        super(CriticPPO, self).__init__()
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, state):
        return self.state_value(state)


class PPODiscreteAgent:
    def __init__(self, state_dim:int, action_dim:int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = 256
        self.actor_lr = 1e-4
        self.critic_lr = 2e-4
        self.entropy_coef= 0.02
        self.max_buffer_size= 500000
        self.batch_size = 1024
        self.clip_epsilon = 0.2
        self.target_step = 4096
        self.repeat_time = 3
        self.reward_scale = 1.
        self.tau = 2 ** -9  # soft update.
        self.gamma = 0.99  # discount factor.
        self.lambda_gae_adv = 0.95  #gae_lambda
        self.if_use_gae = True


        self.net_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = ReplayBuffer(self.state_dim, self.max_buffer_size, self.buffer_device)

        self.actor =  ActorPPO(self.state_dim,self.action_dim, self.mid_dim).to(self.net_device)
        self.critic = CriticPPO(self.state_dim, self.mid_dim).to(self.net_device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.last_state = None
        self.mse_loss = nn.MSELoss()

    @torch.no_grad()
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.net_device)
        action, logprob = self.actor.get_action_and_logprob(state)
        return action.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def explore_env(self, env):
        state = env.reset() if self.last_state is None else self.last_state
        for i in range(self.target_step):
            action, logprob = self.select_action(state)
            state_, reward, done, _ = env.step(int(action))
            others = (reward * self.reward_scale, (1 - done) * self.gamma, action, logprob)
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
            states, actions, reward, mask, old_logprob = self.replay_buffer.sample_all(self.net_device)
            buf_len = len(actions)
            buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.net_device)  # advantage value
            state_values = self.critic(states)
            values = state_values.squeeze(1)
            discount_cumulative_rewards = torch.empty(buf_len, dtype=torch.float32, device=self.net_device)
            tmp_last_state = torch.as_tensor((self.last_state,), dtype=torch.float32, device=self.net_device)
            last_value = self.critic(tmp_last_state)
            for i in range(buf_len - 1, -1, -1):
                discount_cumulative_rewards[i] = reward[i] + mask[i] * last_value
                last_value = discount_cumulative_rewards[i]
            if self.if_use_gae:         #gae
                pre_adv_v = 0  # advantage value of previous step
                for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
                    buf_adv_v[i] = reward[i] + mask[i] * pre_adv_v - values[i]
                    pre_adv_v = values[i] + buf_adv_v[i] * self.lambda_gae_adv
                advantage = buf_adv_v
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
            else:        #reward-to-go
                advantage = discount_cumulative_rewards - (mask * values)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        for _ in range(int(self.target_step * self.repeat_time / self.batch_size)):
            indices = np.random.randint(0, buf_len, self.batch_size)
            batch_state = states[indices]
            batch_action = actions[indices]
            batch_dis_cum_rewards = discount_cumulative_rewards[indices]
            batch_advantage = advantage[indices]
            batch_old_logprob = old_logprob[indices]
            batch_new_logprob, batch_entropy = self.actor.get_logprob_entropy(batch_state, batch_action)
            ratio = (batch_new_logprob - batch_old_logprob).exp()
            surrogate1 = batch_advantage * ratio
            surrogate2 = batch_advantage * ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            actor_loss = obj_surrogate - batch_entropy * self.entropy_coef
            self.optim_update(self.actor_optimizer, actor_loss)
            batch_values = self.critic(batch_state).squeeze(1)
            critic_loss = self.mse_loss(batch_values, batch_dis_cum_rewards)
            self.optim_update(self.critic_optimizer, critic_loss)
        self.replay_buffer.empty_buffer()


    def evaluate(self, env, render=False):
        epochs = 10
        res = np.zeros((epochs,))
        obs = env.reset()
        index = 0
        while index < epochs:
            if render: env.render()
            action, _ = self.select_action(obs)
            s_, reward, done, _ = env.step(int(action))
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














