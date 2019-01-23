import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, occ_obs_shape, sign_obs_shape, state_rep, action_space, recurrent_policy):
        super(Policy, self).__init__()

        if state_rep in ['sign','original']:
            self.base = MLPBase(sign_obs_shape, recurrent_policy)
        elif state_rep == 'full':
            self.base = CNNBase(occ_obs_shape, sign_obs_shape, recurrent_policy)
        else:
            raise NotImplemented('Only implemented sign, origianal, and full state representation')

        num_outputs = action_space.n # 2
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def act(self, occ_inputs, sign_inputs, rnn_hxs, masks, deterministic=False): # Not deterministic, chooses actions wrt output probabilities

        value, actor_features, rnn_hxs = self.base(occ_inputs, sign_inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        #print(action.shape)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, occ_inputs, sign_inputs, rnn_hxs, masks):
        value, _, _ = self.base(occ_inputs, sign_inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, occ_inputs, sign_inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(occ_inputs, sign_inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

# TODO : adapt reccurent to new observations
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class CNNBase(NNBase):
    def __init__(self, occ_num_inputs, sign_num_inputs, recurrent, seperate_lanes = False):

        self.seperate_lanes = seperate_lanes
        combined_size = 4 + sign_num_inputs
        
        super(CNNBase, self).__init__(recurrent, combined_size, combined_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        if self.seperate_lanes:

            self.lane1 = nn.Sequential(
                init_(nn.Conv1d(1,2,6,stride=1)), #(1,125) -> (2,120)
                nn.ReLU(),nn.MaxPool1d(4),                 #(2,120) -> (2,30)
                init_(nn.Conv1d(2,1,6,stride=1)), #(2,30) -> (1,25)
                nn.ReLU(),nn.MaxPool1d(5),                 #(1,25) -> (1,5)
                init_(nn.Linear(5,1))
            )
            self.lane2 = nn.Sequential(
                init_(nn.Conv1d(1,2,6,stride=1)), #(1,125) -> (2,120)
                nn.ReLU(),nn.MaxPool1d(4),                 #(2,120) -> (2,30)
                init_(nn.Conv1d(2,1,6,stride=1)), #(2,30) -> (1,25)
                nn.ReLU(),nn.MaxPool1d(5),                 #(1,25) -> (1,5)
                init_(nn.Linear(5,1))
            )
            self.lane3 = nn.Sequential(
                init_(nn.Conv1d(1,2,6,stride=1)), #(1,125) -> (2,120)
                nn.ReLU(),nn.MaxPool1d(4),                 #(2,120) -> (2,30)
                init_(nn.Conv1d(2,1,6,stride=1)), #(2,30) -> (1,25)
                nn.ReLU(),nn.MaxPool1d(5),                 #(1,25) -> (1,5)
                init_(nn.Linear(5,1))
            )
            self.lane4 = nn.Sequential(
                init_(nn.Conv1d(1,2,6,stride=1)), #(1,125) -> (2,120)
                nn.ReLU(),nn.MaxPool1d(4),                 #(2,120) -> (2,30)
                init_(nn.Conv1d(2,1,6,stride=1)), #(2,30) -> (1,25)
                nn.ReLU(),nn.MaxPool1d(5),                 #(1,25) -> (1,5)
                init_(nn.Linear(5,1))
            )
        else:
            self.lane = nn.Sequential(
                init_(nn.Conv1d(1,2,6,stride=1)), #(1,125) -> (2,120)
                nn.ReLU(),nn.MaxPool1d(4),                 #(2,120) -> (2,30)
                init_(nn.Conv1d(2,1,6,stride=1)), #(2,30) -> (1,25)
                nn.ReLU(),nn.MaxPool1d(5),                 #(1,25) -> (1,5)
                init_(nn.Linear(5,1))
            )

        self.actor = nn.Sequential(
            init_(nn.Linear(combined_size,combined_size)),
            nn.ReLU(),
            init_(nn.Linear(combined_size,combined_size)),
            nn.ReLU()
            )

        self.critic = nn.Sequential(
            init_(nn.Linear(combined_size, combined_size)),
            nn.ReLU(),
            init_(nn.Linear(combined_size, combined_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(combined_size, 1))

        self.train()

    def forward(self, occ_inputs, sign_inputs, rnn_hxs, masks):
        
        if self.seperate_lanes:
            hidden_lanes1 = self.lane1(occ_inputs[:,0,:].unsqueeze(1)).squeeze(1)
            hidden_lanes2 = self.lane2(occ_inputs[:,1,:].unsqueeze(1)).squeeze(1)
            hidden_lanes3 = self.lane3(occ_inputs[:,2,:].unsqueeze(1)).squeeze(1)
            hidden_lanes4 = self.lane4(occ_inputs[:,3,:].unsqueeze(1)).squeeze(1)
            hidden_input = torch.cat((hidden_lanes1, hidden_lanes2, hidden_lanes3, hidden_lanes4, sign_inputs),1)
        else:
            hidden_lanes1 = self.lane(occ_inputs[:,0,:].unsqueeze(1)).squeeze(1)
            hidden_lanes2 = self.lane(occ_inputs[:,1,:].unsqueeze(1)).squeeze(1)
            hidden_lanes3 = self.lane(occ_inputs[:,2,:].unsqueeze(1)).squeeze(1)
            hidden_lanes4 = self.lane(occ_inputs[:,3,:].unsqueeze(1)).squeeze(1)
            hidden_input = torch.cat((hidden_lanes1, hidden_lanes2, hidden_lanes3, hidden_lanes4, sign_inputs),1)

        #if self.is_recurrent:
        #    x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(hidden_input)
        hidden_actor = self.actor(hidden_input)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent):

        hidden_size = num_inputs
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, occ_inputs, sign_inputs, rnn_hxs, masks):
        x = sign_inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
