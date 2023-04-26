import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNet, CriticNet
from common.utils import to_tensor, entropy, index_to_one_hot, index_to_one_hot_tensor, onehot_from_logits


class CAC(Agent):

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

        self.actor = [
            ActorNet(self.state_dim, self.actor_hidden_size,
                     self.action_dim, self.actor_output_act).to(device)
            for _ in range(self.n_agents)]
        self.actor_target = [
            ActorNet(self.state_dim, self.actor_hidden_size,
                     self.action_dim, self.actor_output_act).to(device)
            for _ in range(self.n_agents)]
        self.actor_optimizer = [Adam(self.actor[i].parameters(), lr=self.actor_lr)
                                for i in range(self.n_agents)]

        self.critic = [
            CriticNet(self.global_state_dim, self.n_agents * self.action_dim, self.critic_hidden_size, 1).to(device)
            for _ in range(self.n_agents)]
        self.critic_target = [
            CriticNet(self.global_state_dim, self.n_agents * self.action_dim, self.critic_hidden_size, 1).to(device)
            for _ in range(self.n_agents)]
        self.critic_optimizer = [Adam(self.critic[i].parameters(), lr=self.critic_lr) for i in range(self.n_agents)]

        for i in range(self.n_agents):
            self.critic_target[i].load_state_dict(self.critic[i].state_dict())
            self.actor_target[i].load_state_dict(self.actor[i].state_dict())


    # agent interact with the environment to collect experience
    def interact(self):
        super()._take_one_step()

    # train on a sample batch
    def train(self):
        if self.n_episodes < self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_tensor = to_tensor(batch.states, self.device).view(-1, self.n_agents * self.state_dim)
        actions_index_tensor = to_tensor(batch.actions, self.device, 'int').view(-1, self.n_agents)
        rewards_tensor = to_tensor(batch.rewards, self.device).view(-1, self.n_agents)
        next_states_tensor = to_tensor(batch.next_states, self.device).view(-1, self.n_agents * self.state_dim)
        dones_tensor = to_tensor(batch.dones, self.device).view(-1, self.n_agents)
        global_state_tensor = to_tensor(batch.global_states, self.device).view(-1, self.global_state_dim)
        next_global_state_tensor = to_tensor(batch.next_global_states, self.device).view(-1, self.global_state_dim)

        # partial obs of each agent
        obs = [states_tensor.view(-1, self.n_agents, self.state_dim)[:, i] for i in range(self.n_agents)]
        next_obs = [next_states_tensor.view(-1, self.n_agents, self.state_dim)[:, i] for i in range(self.n_agents)]
        # get joint action
        act = [index_to_one_hot_tensor(actions_index_tensor[:, i], self.action_dim) for i in range(self.n_agents)]
        actions_tensor = th.cat(act, dim=1)
        next_act = [onehot_from_logits(pi(_next_obs)) for pi, _next_obs in zip(self.actor_target, next_obs)]
        next_actions_tensor = th.cat(next_act, dim=1)

        for i in range(self.n_agents):
            current_q = self.critic[i](global_state_tensor, actions_tensor).squeeze(1)
            # compute mean field V(s')
            next_q = self.critic_target[i](next_global_state_tensor, next_actions_tensor).squeeze(1).detach()
            target_q = rewards_tensor[:, i] + self.reward_gamma * next_q * (1. - dones_tensor[:, i])
            # calculate vf loss
            self.critic_optimizer[i].zero_grad()
            if self.critic_loss == "huber":
                vf_loss = nn.SmoothL1Loss()(current_q, target_q)
            else:
                vf_loss = nn.MSELoss()(current_q, target_q)
            vf_loss.backward()

            # compute log action probs and target q
            actions_index = actions_index_tensor[:, i]
            actions_prob = self.actor[i](obs[i])
            log_action_prob = th.log(actions_prob.gather(1, actions_index.unsqueeze(1)).squeeze(1) + 1e-6)
            values = current_q.detach()
            # calculate pg loss and entropy
            self.actor_optimizer[i].zero_grad()
            pg_loss = th.mean(-log_action_prob * values)
            neg_entropy = -th.mean(entropy(actions_prob))

            # loss backward
            actor_loss = pg_loss + self.entropy_reg * neg_entropy
            actor_loss.backward()

            # optimizer step
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic[i].parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor[i].parameters(), self.max_grad_norm)
            self.critic_optimizer[i].step()
            self.actor_optimizer[i].step()

        for i in range(self.n_agents):
            self._soft_update_target(self.critic_target[i], self.critic[i])

    def action(self, state, evaluation=False, eval_records=None):
        state_tensor = to_tensor(state, self.device).view(-1, self.n_agents, self.state_dim)
        actions = np.zeros(self.n_agents, dtype='int')

        for i in range(self.n_agents):
            obs_tensor = state_tensor[:, i]
            actions_prob_tensor = self.actor[i](obs_tensor).squeeze(0)
            actions_list = th.distributions.Categorical(actions_prob_tensor)
            action = actions_list.sample()
            actions[i] = action.item()

        return actions