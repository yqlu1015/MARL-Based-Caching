import copy
import json
import os
import time
from typing import Tuple, List

import numpy as np
from torch import nn
import torch as th

from common.Memory import ReplayMemory
from common.environment.Environment import EdgeMultiAgentEnv
from common.environment.utils.tool import seed_everything

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Agent(object):
    """
    A unified agent interface:
    - interact: interact with the environment to collect experience
        - _take_one_step: take one step
        - _take_n_steps: take n_agents steps
        - _discount_reward: discount roll out rewards
    - train: train on a sample batch
        - _soft_update_target: soft update the target network
    - exploration_action: choose an action based on state with random noise
                            added for exploration in training
    - action: choose an action based on state for execution
    - value: evaluate value for a state-action pair
    - evaluation: evaluation a learned agent
    """

    def __init__(self, env: EdgeMultiAgentEnv, state_dim, action_dim, device='cpu',
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=None, critic_output_act=None,
                 critic_loss="mse", actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 target_tau=0.01, target_update_step=10, max_episodes=2000):

        self.vf_coef = 1
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = env.n_agents
        self.global_state_dim = env.n_models * 2 + env.n_users * env.n_models
        self.env_obs = self.env.reset()
        self.env_state = self.env.get_global_state()
        self.n_episodes = 0
        self.max_episodes = max_episodes
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1
        self.episode_done = False

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_output_act = critic_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.target_tau = target_tau
        self.target_update_step = target_update_step

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = device

        self.mean_actions = None
        self.mean_actions_e = None

        self.actor: List[nn.Module] = []
        self.critic: List[nn.Module] = []
        self.qnet: List[nn.Module] = []
        self.actor_target: List[nn.Module] = []
        self.critic_target: List[nn.Module] = []
        self.qnet_target: List[nn.Module] = []

    # agent interact with the environment to collect experience
    def interact(self):
        pass

    # take one step
    def _take_one_step(self, use_mean=False):
        # if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
        #     self.env_obs = self.env.reset()
        #     self.n_steps = 0
        obs = self.env_obs
        state = self.env_state
        if use_mean:
            action, mean_action = self.mean_action(obs)
            self.mean_actions = mean_action
        else:
            action = self.exploration_action(obs)
        next_obs, reward, done, info = self.env.step(action)
        self.env_obs = next_obs
        next_state = info[2]
        self.env_state = next_state

        if done[0]:
            if self.done_penalty is not None:
                reward = np.ones(self.n_agents) * self.done_penalty
            # next_state = np.zeros_like(state)
            # self.env_obs = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
            self.env_obs = self.env.reset()
            self.env_state = self.env.get_global_state()
        else:
            self.episode_done = False
        # self.n_steps += 1
        if use_mean:
            self.memory.push(obs, action, reward, next_obs, done, mean_actions=mean_action, global_states=state,
                             next_global_states=next_state)
        else:
            self.memory.push(obs, action, reward, next_obs, done, global_states=state, next_global_states=next_state)

    # take n_agents steps
    # no latest implementation
    # def _take_n_steps(self):
    #     if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
    #         self.env_obs = self.env.reset()
    #         self.n_steps = 0
    #     states = []
    #     actions = []
    #     rewards = []
    #     # take n_agents steps
    #     for i in range(self.roll_out_n_steps):
    #         states.append(self.env_obs)
    #         action = self.exploration_action(self.env_obs)
    #         next_state, reward, done, _ = self.env.step(action)
    #         next_state = next_state
    #         actions.append(action)
    #         if done[0] and self.done_penalty is not None:
    #             reward = np.ones(self.n_agents) * self.done_penalty
    #         rewards.append(reward)
    #         final_state = next_state
    #         self.env_obs = next_state
    #         if done[0]:
    #             self.env_obs = self.env.reset()
    #             break
    #     # discount reward
    #     if done[0]:
    #         final_value = np.zeros(self.n_agents)
    #         self.n_episodes += 1
    #         self.episode_done = True
    #     else:
    #         self.episode_done = False
    #         final_action = self.action(final_state)
    #         final_value = self.value(final_state, final_action)
    #     rewards = self._discount_reward(rewards, final_value)
    #     self.n_steps += 1
    #     self.memory.push(states, actions, rewards)

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the qnet target network or qnet target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    # train on a sample batch
    def train(self):
        pass

    # get actions and mean actions by alternatively updating policy and mean action
    def mean_action(self, state) -> Tuple[np.ndarray, np.ndarray]:
        pass

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state) -> np.ndarray:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * (self.n_episodes - self.episodes_before_train) / self.epsilon_decay)
        # epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #           ((self.max_episodes - self.n_episodes) / (self.max_episodes - self.episodes_before_train))
        if self.n_episodes < self.episodes_before_train or np.random.rand() < epsilon:
            action = np.random.choice(self.env.n_actions, self.env.n_agents)
        else:
            action = self.action(state, evaluation=False)
        return action

    # choose an action based on state for execution
    def action(self, state, evaluation=False, eval_records=None) -> np.ndarray:
        pass

    # evaluate value for a state-action pair
    def value(self, state, action) -> np.ndarray:
        pass

    # evaluation the learned agent
    def evaluation(self, env: EdgeMultiAgentEnv, eval_episodes=10, seed=0):
        rewards = []
        infos = []
        rewards_qoe = []  # average qoe at edge
        rewards_switch = []
        users_info = []
        stats = {'cache': [], 'delay': [], 'ratio': [], 'switch': [], 'qoe': []}
        eval_records = []  # top 3 actor values and indices
        # seed_everything(seed)

        for i in range(eval_episodes):
            self.mean_actions_e = np.zeros((self.n_agents, self.action_dim))
            rewards_i = []
            rewards_qoe_i = []
            rewards_switch_i = []
            state = env.reset()
            # action = self.action(state)
            # state, reward, done, info = env.step(action)
            #
            # done = done[0] if isinstance(done, np.ndarray) else done
            # rewards_i.append(reward)
            # rewards_qoe_i.append(info[0])
            # rewards_switch_i.append(info[1])
            # stats['delay'].append(env.world.average_delay)
            # stats['ratio'].append(env.world.cache_hit_ratio)
            # stats['cache'].append(env.world.cache_stat)
            # stats['switch'].append(env.world.switch_sum)
            # delays = []
            # switches = []
            # for agent in env.world.agents:
            #     delays.append(agent.avg_qoe)
            #     switches.append(agent.switch)
            # stats['agent_delay'].append(delays)
            # stats['agent_switch'].append(switches)

            done = False
            while not done:
                action = self.action(state, evaluation=True, eval_records=eval_records)
                # action = self.exploration_action(state)
                state, reward, done, info = env.step(action)

                done = done[0] if isinstance(done, np.ndarray) else done
                rewards_i.append(reward)
                rewards_qoe_i.append(info[0])
                rewards_switch_i.append(info[1])
                stats['delay'].append(env.world.average_delay)
                stats['ratio'].append(env.world.cache_hit_ratio)
                stats['cache'].append(env.world.cache_stat)
                stats['switch'].append(env.world.switch_sum)
                stats['qoe'].append(env.world.avg_qoe)

            rewards.append(rewards_i)
            rewards_qoe.append(rewards_qoe_i)
            rewards_switch.append(rewards_switch_i)
            infos.append(env.state_info())
            users_info.append(env.users_info())
            stats['eval_records'] = eval_records

        return rewards, infos, rewards_qoe, rewards_switch, users_info, stats

    def save_models(self, algo_id: str):
        time_str = time.strftime('%m_%d_%H_%M')
        if self.actor:
            for i, (actor, actor_target) in enumerate(zip(self.actor, self.actor_target)):
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_actor_{i}.pt"
                th.save(actor.state_dict(), path)
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_actor_target_{i}.pt"
                th.save(actor_target.state_dict(), path)

        if self.critic and not isinstance(self.critic, nn.Module):
            for i, (critic, critic_target) in enumerate(zip(self.critic, self.critic_target)):
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_critic_{i}.pt"
                th.save(critic.state_dict(), path)
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_critic_target_{i}.pt"
                th.save(critic_target.state_dict(), path)

        if self.critic and isinstance(self.critic, nn.Module):
            path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_critic.pt"
            path1 = f"{ROOT_DIR}/models/{time_str}_{algo_id}_critic_target.pt"
            th.save(self.critic.state_dict(), path)
            th.save(self.critic_target.state_dict(), path1)

        if self.qnet:
            for i, (actor, actor_target) in enumerate(zip(self.qnet, self.qnet_target)):
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_qnet_{i}.pt"
                th.save(actor.state_dict(), path)
                path = f"{ROOT_DIR}/models/{time_str}_{algo_id}_qnet_target_{i}.pt"
                th.save(actor_target.state_dict(), path)

    def load_models(self, paths: dict):
        if paths['actor']:
            for i, (actor, actor_target) in enumerate(zip(self.actor, self.actor_target)):
                path = paths['actor'][i]
                actor.load_state_dict(th.load(path))
                path = path['actor_target'][i]
                actor_target.load_state_dict(th.load(path))

        if paths['critic'] and not isinstance(paths['critic'], str):
            for i, (critic, critic_target) in enumerate(zip(self.critic, self.critic_target)):
                path = paths['critic'][i]
                critic.load_state_dict(th.load(path))
                path = paths['critic_target'][i]
                critic_target.load_state_dict(th.load(path))

        if paths['critic'] and isinstance(paths['critic'], str):
            path = paths['critic']
            path1 = paths['critic_target']
            self.critic.load_state_dict(th.load(path))
            self.critic_target.load_state_dict(th.load(path1))

        if paths['qnet']:
            for i, (actor, actor_target) in enumerate(zip(self.qnet, self.qnet_target)):
                path = paths['qnet'][i]
                actor.load_state_dict(th.load(path))
                path = paths['qnet_target'][i]
                actor_target.load_state_dict(th.load(path))

    def save_memory(self, algo_id: str):
        time_str = time.strftime('%m_%d_%H_%M')
        with open(f'{ROOT_DIR}/memory/{time_str}/{algo_id}', 'w') as fp:
            json.dump(self.memory, fp)

    def load_memory(self, path):
        with open(path, 'r') as fp:
            self.memory = json.load(fp)
