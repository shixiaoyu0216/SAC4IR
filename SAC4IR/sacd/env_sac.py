import numpy as np


class Env():
    def __init__(self, observation_data, I, max_item_id, each_user, K, item_ctr_dict, pop_dict):
        self.observation = np.array(observation_data)
        self.n_observation = len(self.observation)
        self.I = I
        self.action_space = range(1, max_item_id + 1)
        self.n_actions = max_item_id
        self.user = each_user
        self.K = K
        self.item_ctr_dict = item_ctr_dict
        self.item_ips_dict = item_ctr_dict
        self.pop_dict = pop_dict
        self.rec_list = K * 2

    def reset(self, observation):
        self.observation = observation
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action, pass_item_list):
        done = False
        s = self.observation
        r = 0

        if s[-1] == action:
            r = -1
        elif action not in self.I:
            r = -1
        else:
            r_pop = 0
            r_ctr = 0
            for i in range(self.n_observation):
                if str(action) in self.item_ctr_dict[str(s[-(i + 1)])].keys():
                    r_ctr += self.item_ctr_dict[str(s[-(i + 1)])][str(action)]
            alpha = 1.0
            beta = 0.4
            if action in pass_item_list:
                alpha = -2.0
            r = alpha * r_ctr + beta * r_pop

        if r > 0:
            s_temp_ = np.append(s, action)
            observation_ = np.delete(s_temp_, 0, axis=0)
            # done = True
        else:
            observation_ = s
        s_ = observation_
        return s_, r, done
