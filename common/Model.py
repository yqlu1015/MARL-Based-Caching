from typing import Optional

import torch as th
from torch import nn


class ActorNet(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, mid_dim, output_size, output_act):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, output_size),
        )
        # self.fc1 = nn.Linear(state_dim + action_dim, mid_dim).double()
        # self.fc2 = nn.Linear(mid_dim, mid_dim).double()
        # self.fc3 = nn.Linear(mid_dim, output_size).double()
        # activation function for the output
        self.output_act = output_act

    def forward(self, state: th.Tensor, mask: Optional[th.Tensor] = None):
        out = self.net(state)
        out = self.output_act(out)
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ValueNet(nn.Module):
    """
    A network for value function
    """

    def __init__(self, state_dim, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorCriticNetwork(nn.Module):
    """
    An qnet-qnet network that shared lower-layer representations but
    have distinct output layers
    """

    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val


class MeanCriticNet(nn.Module):
    """
    A network for mf-ac with state, action and mean action as input
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size, output_act):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim + 1 + action_dim, mid_dim),
        #     nn.ReLU(),
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.ReLU(),
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.ReLU(),
        #     nn.Linear(mid_dim, output_size),
        # ).double()
        self.fc1 = nn.Linear(state_dim, mid_dim)
        self.fc2 = nn.Linear(action_dim + mid_dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()
        self.output_act = output_act

    def forward(self, state, mean_action):
        out = self.activate(self.fc1(state))
        out = th.cat((out, mean_action), 1)
        out = self.activate(self.fc2(out))
        out = self.fc3(out)
        out = self.output_act(out)
        return out


class MeanQNet(nn.Module):
    """
    A network for mf-q with both state and mean action as input
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size, output_act):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim + action_dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()
        self.output_act = output_act

    def forward(self, state, mean_action):
        out = self.activate(self.fc1(state))
        # actions = self.activate(self.fc2(mean_action))
        out = th.cat((out, mean_action), 1)
        out = self.activate(self.fc2(out))
        out = self.fc3(out)
        out = self.output_act(out)
        return out
