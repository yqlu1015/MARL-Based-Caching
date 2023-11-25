import torch as th
from torch import nn
from torch.optim import Adam
import numpy as np

from common.Agent import Agent
from common.Model import ActorNet
from common.utils import to_tensor


class DQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """

    def __init__(self, env, state_dim, action_dim, device='cpu',
                 memory_capacity=10000, max_steps=10000, max_episodes=400,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=64, critic_hidden_size=64,
                 actor_output_act=None, critic_output_act=None,
                 critic_loss="mse", actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=1000, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 target_tau=0.01, target_update_step=10, seed=0):
        super().__init__(env, state_dim, action_dim, device, memory_capacity, max_steps, reward_gamma, reward_scale,
                         done_penalty, actor_hidden_size, critic_hidden_size, actor_output_act, critic_output_act,
                         critic_loss, actor_lr, critic_lr, optimizer_type, entropy_reg, max_grad_norm, batch_size,
                         episodes_before_train, epsilon_start, epsilon_end, epsilon_decay, target_tau,
                         target_update_step, max_episodes)

        self.qnet = [ActorNet(self.state_dim, self.actor_hidden_size,
                              self.action_dim, self.critic_output_act).to(device)
                     for i in range(self.n_agents)]
        self.qnet_target = [ActorNet(self.state_dim, self.actor_hidden_size,
                                     self.action_dim, self.critic_output_act).to(device)
                            for i in range(self.n_agents)]
        for i in range(self.n_agents):
            self.qnet_target[i].load_state_dict(self.qnet[i].state_dict())
        self.qnet_optimizer = [Adam(self.qnet[i].parameters(), lr=self.critic_lr)
                               for i in range(self.n_agents)]

    # agent interact with the environment to collect experience
    def interact(self):
        super()._take_one_step()

    # train on a sample batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_tensor = to_tensor(batch.states, self.device).view(-1, self.n_agents, self.state_dim)
        actions_tensor = to_tensor(batch.actions, self.device, 'int').view(-1, self.n_agents)
        rewards_tensor = to_tensor(batch.rewards, self.device).view(-1, self.n_agents)
        next_states_tensor = to_tensor(batch.next_states, self.device).view(-1, self.n_agents, self.state_dim)
        dones_tensor = to_tensor(batch.dones, self.device).view(-1, self.n_agents)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        for i in range(self.n_agents):

            current_q = self.qnet[i](states_tensor[:, i]).gather(1, actions_tensor[:, i].unsqueeze(1)).squeeze(1)

            # compute V(s_{t+1}) for all next states and all actions,
            # and we then take max_a { V(s_{t+1}) }
            next_state_action_values = self.qnet_target[i](next_states_tensor[:, i]).detach()
            next_q = th.max(next_state_action_values, 1)[0].view(-1)
            # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            target_q = self.reward_scale * rewards_tensor[:, i] + self.reward_gamma * next_q * (1. - dones_tensor[:, i])

            # update value network
            self.qnet_optimizer[i].zero_grad()
            if self.critic_loss == "huber":
                loss = nn.SmoothL1Loss()(current_q, target_q)
            else:
                loss = nn.MSELoss()(current_q, target_q)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.qnet[i].parameters(), self.max_grad_norm)
            self.qnet_optimizer[i].step()

        for i in range(self.n_agents):
            self._soft_update_target(self.qnet_target[i], self.qnet[i])

    # choose an action based on state for execution
    def action(self, state, evaluation=False, eval_records=None):
        action = np.zeros(self.n_agents, dtype='int')
        state_tensor = to_tensor(state, self.device).view(-1, self.n_agents, self.state_dim)
        for i in range(self.n_agents):
            state_action_values_tensor = self.qnet[i](state_tensor[:, i]).squeeze(0)
            action[i] = th.argmax(state_action_values_tensor, dim=0).item()
            if evaluation and i == 0:
                state_action_values_tensor = th.softmax(state_action_values_tensor, dim=0)
                max_values, indices = th.topk(state_action_values_tensor, 3)
                row = np.hstack((max_values.detach().cpu().numpy(), indices.detach().cpu().numpy()))
                eval_records.append(row)
        return action
