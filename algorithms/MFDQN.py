from typing import Tuple

import torch as th
from torch import nn
from torch.optim import Adam

import numpy as np

from common.Agent import Agent
from common.Model import MeanQNet
from common.utils import to_tensor, index_to_one_hot, sample_from_logits, index_to_one_hot_tensor, onehot_from_logits


class MFDQN(Agent):

    def __init__(self, env, state_dim, action_dim, device='cpu',
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=64, critic_hidden_size=64,
                 actor_output_act=None, critic_output_act=None,
                 critic_loss="mse", actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=1000, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 target_tau=0.01, target_update_step=10, max_episodes=1000):
        super().__init__(env, state_dim, action_dim, device, memory_capacity, max_steps, reward_gamma, reward_scale,
                         done_penalty, actor_hidden_size, critic_hidden_size, actor_output_act, critic_output_act,
                         critic_loss, actor_lr, critic_lr, optimizer_type, entropy_reg, max_grad_norm, batch_size,
                         episodes_before_train, epsilon_start, epsilon_end, epsilon_decay, target_tau,
                         target_update_step, max_episodes)

        self.temperature = self.env.temperature

        self.qnet = [
            MeanQNet(self.state_dim, self.action_dim, self.actor_hidden_size, self.action_dim,
                     self.critic_output_act).to(device)
            for i in range(self.n_agents)]
        self.qnet_target = [
            MeanQNet(self.state_dim, self.action_dim, self.actor_hidden_size,
                     self.action_dim, self.critic_output_act).to(device)
            for i in range(self.n_agents)]
        for i in range(self.n_agents):
            self.qnet_target[i].load_state_dict(self.qnet[i].state_dict())
        self.qnet_optimizer = [Adam(self.qnet[i].parameters(), lr=self.critic_lr) for i in range(self.n_agents)]

        self.mean_actions = np.zeros((self.n_agents, self.env.n_actions))
        self.mean_actions_e = np.zeros((self.n_agents, self.env.n_actions))

    # agent interact with the environment to collect experience
    def interact(self):
        super()._take_one_step(use_mean=True)

    # train on a sample batch
    def train(self):
        if self.n_episodes < self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_tensor = to_tensor(batch.states, self.device).view(-1, self.n_agents, self.state_dim)
        actions_tensor = to_tensor(batch.actions, self.device, 'int').view(-1, self.n_agents)
        rewards_tensor = to_tensor(batch.rewards, self.device).view(-1, self.n_agents)
        next_states_tensor = to_tensor(batch.next_states, self.device).view(-1, self.n_agents, self.state_dim)
        dones_tensor = to_tensor(batch.dones, self.device).view(-1, self.n_agents)
        mean_actions_tensor = to_tensor(batch.mean_actions, self.device).view(-1, self.n_agents, self.action_dim)

        obs = [states_tensor[:, i] for i in range(self.n_agents)]
        next_obs = [next_states_tensor[:, i] for i in range(self.n_agents)]
        next_act = [onehot_from_logits(pi(_next_obs, mean_actions_tensor[:, i]))
                    for i, (pi, _next_obs) in enumerate(zip(self.qnet, next_obs))]

        # compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        for i in range(self.n_agents):
            actions_index = actions_tensor[:, i].unsqueeze(1)
            current_q = self.qnet[i](obs[i], mean_actions_tensor[:, i]).gather(1, actions_index).squeeze(1)

            # calculate next mean action
            neighbors = self.env.get_obs(i)
            next_mean_action_tensor = th.zeros_like(mean_actions_tensor[:, i], device=self.device)
            for j in neighbors:
                next_mean_action_tensor += next_act[j]
            next_mean_action_tensor /= len(neighbors)

            # compute mean field V(s')

            # next_actions_index = th.argmax(next_state_action_values_tensor, dim=1).unsqueeze(1)
            # next_v = self.qnet_target(next_states_tensor[:, i],
            #                           mean_actions_tensor[:, i]).gather(1, next_actions_index).squeeze(1).detach()
            next_q = th.sum(self.qnet_target[i](next_obs[i], next_mean_action_tensor) * next_act[i], dim=1).detach()

            # next_v = th.einsum('bj,bj->b', next_state_action_values, action_probs)
            # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            target_q = rewards_tensor[:, i] + self.reward_gamma * next_q * (1. - dones_tensor[:, i])

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

    # get actions and mean actions by alternatively updating policy and mean action
    def mean_action(self, state) -> Tuple[np.ndarray, np.ndarray]:
        state_tensor = to_tensor(state, self.device).view(-1, self.n_agents, self.state_dim)
        actions = np.zeros(self.n_agents, dtype='int')
        mean_actions = self.mean_actions
        mean_actions_tensor = to_tensor(mean_actions, self.device).view(-1, self.n_agents, self.action_dim)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * (self.n_episodes - self.episodes_before_train) / self.epsilon_decay)

        # update policies
        for i in range(self.n_agents):
            state_action_values_tensor = self.qnet[i](state_tensor[:, i], mean_actions_tensor[:, i]).squeeze(0)
            if self.n_episodes < self.episodes_before_train or np.random.rand() < epsilon:
                actions[i] = np.random.choice(self.env.n_actions)
            else:
                # probs = th.softmax(state_action_values_tensor / self.temperature, dim=0)
                # actions_list = th.distributions.Categorical(probs)
                # action = actions_list.sample()
                # actions[i] = action.item()
                actions[i] = th.argmax(state_action_values_tensor, dim=0).item()

        # update mean actions
        for i in range(self.n_agents):
            neighbors = self.env.get_obs(i)
            neighbor_actions = []
            for j in neighbors:
                action = index_to_one_hot(actions[j], self.action_dim)
                neighbor_actions.append(action)

                mean_actions[i] = np.mean(neighbor_actions, axis=0)

        return actions, mean_actions.copy()

    def action(self, state, evaluation=False, eval_records=None):
        state_tensor = to_tensor(state, self.device).view(-1, self.n_agents, self.state_dim)
        actions = np.zeros(self.n_agents, dtype='int')
        mean_actions = self.mean_actions_e if evaluation else self.mean_actions
        mean_actions_tensor = to_tensor(mean_actions, self.device).view(-1, self.n_agents, self.action_dim)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * (self.n_episodes - self.episodes_before_train) / self.epsilon_decay)

        # update policies
        for i in range(self.n_agents):
            state_action_values_tensor = self.qnet[i](state_tensor[:, i], mean_actions_tensor[:, i]).squeeze(0)
            if evaluation:
                actions[i] = th.argmax(state_action_values_tensor, dim=0).item()
                if i == 0:
                    state_action_values_tensor = th.softmax(state_action_values_tensor, dim=0)
                    max_values, indices = th.topk(state_action_values_tensor, 3)
                    row = np.hstack((max_values.detach().cpu().numpy(), indices.detach().cpu().numpy()))
                    eval_records.append(row)
            elif self.n_episodes < self.episodes_before_train or np.random.rand() < epsilon:
                actions[i] = np.random.choice(self.env.n_actions)
            else:
                # probs = th.softmax(state_action_values_tensor / self.temperature, dim=0)
                # actions_list = th.distributions.Categorical(probs)
                # action = actions_list.sample()
                # actions[i] = action.item()
                actions[i] = th.argmax(state_action_values_tensor, dim=0).item()

        # update mean actions
        for i in range(self.n_agents):
            neighbors = self.env.get_obs(i)
            neighbor_actions = []
            for j in neighbors:
                action = index_to_one_hot(actions[j], self.action_dim)
                neighbor_actions.append(action)

                mean_actions[i] = np.mean(neighbor_actions, axis=0)

        return actions
