from typing import Optional

import torch as th
from torch import nn


class ActorNet(nn.Module):
    """
    A network for qnet
    """

    def __init__(self, state_dim, mid_dim, output_size, output_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            # nn.Linear(mid_dim, mid_dim),
            # nn.ReLU(),
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


class CriticNet(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size=1):
        super().__init__()
        self.fc0 = nn.Linear(action_dim, mid_dim)
        self.fc1 = nn.Linear(state_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim * 2, mid_dim)
        # self.fc4 = nn.Linear(mid_dim * 2, mid_dim * 2)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()

    def forward(self, state, action):
        action_dense = self.activate(self.fc0(action))
        state_dense = self.activate(self.fc1(state))
        out = th.cat((state_dense, action_dense), 1)
        out = self.activate(self.fc2(out))
        # out = self.activate(self.fc4(out))
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
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        # out = nn.functional.relu(self.fc4(out))
        out = self.fc3(out)
        return out


class MeanValueNet(nn.Module):
    """
    A network for mf-ac with state, action and mean action as input
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size, output_act):
        super().__init__()
        self.fc0 = nn.Linear(action_dim, mid_dim)
        self.fc1 = nn.Linear(state_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim * 2, mid_dim)
        # self.fc4 = nn.Linear(mid_dim * 2, mid_dim * 2)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()
        self.output_act = output_act

    def forward(self, state, mean_action):
        action_dense = self.activate(self.fc0(mean_action))
        state_dense = self.activate(self.fc1(state))
        out = th.cat((state_dense, action_dense), 1)
        out = self.activate(self.fc2(out))
        # out = self.activate(self.fc4(out))
        out = self.fc3(out)
        out = self.output_act(out)
        return out


class MeanQNet(nn.Module):
    """
    A network for mf-q with both state and mean action as input
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size, output_act):
        super().__init__()
        self.fc0 = nn.Linear(action_dim, mid_dim)
        self.fc1 = nn.Linear(state_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim * 2, mid_dim)
        # self.fc4 = nn.Linear(mid_dim * 2, mid_dim * 2)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()
        self.output_act = output_act

    def forward(self, state, mean_action):
        action_dense = self.activate(self.fc0(mean_action))
        state_dense = self.activate(self.fc1(state))
        out = th.cat((state_dense, action_dense), 1)
        out = self.activate(self.fc2(out))
        # out = self.activate(self.fc4(out))
        out = self.fc3(out)
        out = self.output_act(out)
        return out


class MeanCriticNet(nn.Module):
    """
    A network for mf-ac with state, action and mean action as input
    """

    def __init__(self, state_dim, action_dim, mid_dim, output_size, output_act):
        super().__init__()
        self.fc0 = nn.Linear(action_dim, mid_dim)
        self.fc1 = nn.Linear(state_dim + action_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim * 2, mid_dim)
        self.fc3 = nn.Linear(mid_dim, output_size)
        self.activate = nn.ReLU()
        self.output_act = output_act

    def forward(self, state, action, mean_action):
        action_dense = self.activate(self.fc0(mean_action))
        state_action = th.cat((state, action), 1)
        state_dense = self.activate(self.fc1(state_action))
        out = th.cat((state_dense, action_dense), 1)
        out = self.activate(self.fc2(out))
        out = self.fc3(out)
        out = self.output_act(out)
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
