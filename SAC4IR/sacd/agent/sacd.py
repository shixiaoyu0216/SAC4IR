import os
import numpy as np
import torch
from torch.optim import Adam
from .base import BaseAgent
from sacd.model import TwinnedQNetwork, PolicyNetwork, BaseNetwork
from sacd.utils import disable_gradients, update_params


class SacdAgent(BaseAgent):
    def __init__(self, env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=False, state_re=True, seed=0, K=10):
        super().__init__(
            env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, state_re, seed, K)

        self.actor = PolicyNetwork(self.env.n_observation, self.env.n_actions, self.env, self.batch_size,
                                   state_representation=state_re).to(self.device)
        self.main_critic = TwinnedQNetwork(self.env.n_observation, self.env.n_actions, self.batch_size,
                                           state_representation=state_re, dueling_net=dueling_net).to(
            device=self.device)
        self.target_critic = TwinnedQNetwork(self.env.n_observation, self.env.n_actions, self.batch_size,
                                             state_representation=state_re, dueling_net=dueling_net).to(
            device=self.device).eval()
        self.target_critic.load_state_dict(self.main_critic.state_dict())
        disable_gradients(self.target_critic)
        self.policy_optim = Adam(self.actor.parameters(), lr=lr)
        self.q1_optim = Adam(self.main_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.main_critic.Q2.parameters(), lr=lr)
        self.avg_prob = 1.0 / self.env.n_actions
        self.maximum_entropy = -np.log(self.avg_prob) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    # 探索 随机动作
    def explore(self, state):
        state = np.array([int(i) for i in state[0].split('|')])
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return action.item()

    def exploit(self, state, user_id):
        s = self.env.reset(np.array([int(i) for i in state.split('|')]))
        s = torch.from_numpy(s).to(self.device)
        with torch.no_grad():
            action = self.actor.act(s, user_id)
        return action

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.main_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.main_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long() - 1)
        curr_q2 = curr_q2.gather(1, actions.long() - 1)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (action_probs * (next_q - self.alpha * log_action_probs)).sum(dim=1, keepdim=True)
        return rewards + self.gamma_n * next_v

    def calc_critic_loss(self, batch):
        target_q = self.calc_target_q(*batch)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        errors = torch.abs(curr_q1.detach() - target_q)

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q).pow(2))

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        _, action_probs, log_action_probs = self.actor.sample(states)
        with torch.no_grad():
            q1, q2 = self.main_critic(states)
            q = torch.min(q1, q2)

        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
        q = torch.sum(action_probs * q, dim=1, keepdim=True)
        policy_loss = -(self.alpha * entropies + q).mean()
        update_params(self.policy_optim, policy_loss)

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad
        entropy_loss = torch.mean(self.log_alpha * (self.maximum_entropy - entropies))
        update_params(self.alpha_optim, entropy_loss)
        self.alpha = self.log_alpha.exp()
        return self.alpha

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.actor.save(os.path.join(save_dir, 'policy.pth'))
        self.main_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
