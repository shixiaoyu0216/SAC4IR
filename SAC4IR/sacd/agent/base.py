from abc import ABC, abstractmethod
import os
import numpy as np
import torch

from sacd.memory import LazyMultiStepMemory


class BaseAgent(ABC):
    def __init__(self, env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=False, state_representation=True, seed=0, K=10):
        super().__init__()

        self.env = env
        if cuda is True and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.memory = LazyMultiStepMemory(
            capacity=memory_size,
            state_shape=self.env.n_observation,
            device=self.device, gamma=gamma, multi_step=multi_step)

        self.steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # 损失值列表，画图用
        self.policy_loss_list = []
        self.alpha_value_list = []
        self.q1_loss_list = []
        self.q2_loss_list = []
        self.q1_mean_list = []
        self.q2_mean_list = []
        self.td_errors_list = []

    def run_offpolicy(self, history_list):
        self.train_episode_offpolicy(history_list)

    def run_onpolicy(self, user_history, train_history_idx):
        learn_success = self.train_episode_onpolicy(user_history, train_history_idx)
        return learn_success

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state, user_id):
        pass

    @abstractmethod
    def update_target_critic(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies):
        pass

    def train_episode_onpolicy(self, user_history, train_history_idx):
        done = False
        history = user_history[train_history_idx - 1]
        s = np.array([int(i) for i in history[0].split('|')])
        a = np.array(int(history[1]))
        r = np.array(float(history[2]))
        s_ = np.array([int(i) for i in history[3].split('|')])
        # r = max(min(r, 1.0), -1.0)
        self.memory.append(s, a, r, s_, done)

        episode_steps = 0
        while True:
            if self.memory.__len__() >= 10:
                self.learn()
                if episode_steps % self.target_update_interval == 0:
                    self.update_target_critic()
                    episode_steps += 1
                elif episode_steps == self.max_episode_steps:
                    return True
                else:
                    episode_steps += 1
            else:
                return False

    def train_episode_offpolicy(self, history_list):
        done = False
        for history in history_list:
            s = np.array([int(i) for i in history[0].split('|')])
            a = np.array(int(history[1]))
            r = np.array(float(history[2]))
            s_ = np.array([int(i) for i in history[3].split('|')])
            # r = max(min(r, 1.0), -1.0) # 奖励归一化
            self.memory.append(s, a, r, s_, done)
        episode_steps = 0
        while True:
            if self.memory.__len__() >= 10:
                self.learn()
                if episode_steps % self.target_update_interval == 0:
                    self.update_target_critic()
                    episode_steps += 1
                elif episode_steps == self.max_episode_steps:
                    break
                else:
                    episode_steps += 1

    def learn(self):
        batch = self.memory.sample(self.batch_size)
        policy_loss, entropies = self.calc_policy_loss(batch)
        alpha = self.calc_entropy_loss(entropies)
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch)

        self.policy_loss_list.append(policy_loss.cpu().item())
        self.alpha_value_list.append(alpha.cpu().item())
        self.q1_loss_list.append(q1_loss.cpu().item())
        self.q2_loss_list.append(q2_loss.cpu().item())
        self.q1_mean_list.append(mean_q1)
        self.q2_mean_list.append(mean_q2)
        # self.td_errors_list.append(errors.cpu().item())

    # 模型存储
    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        pass
