import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent


class Base(Agent):
    """
    An agent randomly choose actions from the action space as a baseline algorithm
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
                         target_update_step)

    # agent interact with the environment to collect experience
    def interact(self):
        super()._take_one_step()

    def action(self, state) -> np.ndarray:
        action = np.random.choice(self.env.n_actions, self.n_agents)
        return action

    def exploration_action(self, state) -> np.ndarray:
        return self.action(state)
