import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StateEncodeNet(BaseNetwork):
    def __init__(self, env_n_actions, embedding_dim=10, hidden_size=10):
        super(StateEncodeNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder_hidden_size = hidden_size
        self.embedding = nn.Embedding(env_n_actions + 1, self.embedding_dim)
        self.mapping_qk = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mapping_v = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.atten = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.embedding_dim, batch_first=True)
        self.norm1 = nn.Linear(self.embedding_dim, 1)
        self.norm2 = nn.Linear(self.embedding_dim, 1)

    def forward(self, states, hidden_size=10):
        self.hidden_size = hidden_size
        embed_states = self.embedding(states.long())
        qk = self.mapping_qk(embed_states)
        v = self.mapping_v(embed_states)
        attention_out, attention_weight = self.atten(query=qk, key=qk, value=v)
        norm_out = self.norm1(attention_out)
        norm_out = norm_out.view([self.hidden_size, -1])
        norm_out = self.norm2(norm_out)
        return norm_out


class QNetwork(BaseNetwork):
    def __init__(self, num_states, num_actions, batch_size, state_representation=False,
                 dueling_net=False):
        super().__init__()
        self.num_states = num_states
        self.batch_size = batch_size
        self.state_representation = state_representation
        self.dueling_net = dueling_net
        if self.state_representation:
            self.re_state = StateEncodeNet(num_actions)
            self.num_states = 1
        if dueling_net:
            self.a_head = nn.Sequential(
                nn.Linear(self.num_states, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(self.num_states, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1))
        else:
            self.q_head = nn.Sequential(
                nn.Linear(self.num_states, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_actions))

    def forward(self, states):
        if self.state_representation:
            states_re = self.re_state(states, hidden_size=self.batch_size)
        else:
            states_re = states

        if self.dueling_net:
            a = self.a_head(states_re.float())
            v = self.v_head(states_re.float())
            return v + a - a.mean(1, keepdim=True)
        else:
            return self.q_head(states_re.float())


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_states, num_actions, batch_size, state_representation=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_states, num_actions, batch_size, state_representation, dueling_net)
        self.Q2 = QNetwork(num_states, num_actions, batch_size, state_representation, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class PolicyNetwork(BaseNetwork):
    def __init__(self, num_states, num_actions, env, batch_size, state_representation=False):
        super().__init__()
        self.env = env
        self.num_states = num_states
        self.batch_size = batch_size
        self.state_representation = state_representation
        if self.state_representation:
            self.re_state = StateEncodeNet(num_actions)
            self.num_states = 1

        self.q_head = nn.Sequential(
            nn.Linear(self.num_states, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions))

    def act(self, states, user_id):
        if self.state_representation:
            states_re = self.re_state(states.view(1, -1), hidden_size=1)
        else:
            states_re = states
        last_s = states[-1].cpu().item()
        action_q = torch.abs(self.q_head(states_re.squeeze(0).float()))
        action_ctr = torch.clone(action_q)
        for i in range(1, len(action_q) + 1):
            if str(i) in self.env.item_ctr_dict[str(last_s)].keys():
                ctr_value = self.env.item_ctr_dict[str(last_s)][str(i)]
            else:
                ctr_value = 0
            action_ctr[i - 1] = ctr_value
        action_q = action_q * action_ctr
        action_list = []
        while (True):
            greedy_actions = torch.argmax(action_q, dim=0, keepdim=True)
            greedy_actions_id = greedy_actions.cpu().numpy()[0] + 1
            if greedy_actions_id in self.env.observation:
                action_q[greedy_actions] = torch.min(action_q)
                continue
            if greedy_actions_id not in self.env.I:
                action_q[greedy_actions] = torch.min(action_q)
                continue
            action_list.append(greedy_actions_id)
            action_q[greedy_actions] = torch.min(action_q)
            if len(action_list) == self.env.rec_list:
                return action_list

    def sample(self, states):
        if self.state_representation:
            states_re = self.re_state(states, hidden_size=self.batch_size)
        else:
            states_re = states
        action_q = self.q_head(states_re.float())
        action_q = ((action_q - torch.min(action_q)) * 2.0) / (torch.max(action_q) - torch.min(action_q)) - 1
        action_prob = F.softmax(action_q, dim=1)
        action_dist = Categorical(action_prob)
        actions = action_dist.sample().view(-1, 1)
        z = (action_prob == 0.0).float() * 1e-8
        log_action_prob = torch.log(action_prob + z)
        return actions, action_prob, log_action_prob
