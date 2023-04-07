import copy

import numpy as np

from common.Memory import ReplayMemory
from common.environment.Environment import EdgeMultiAgentEnv
from common.environment.utils.tool import seed_everything


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
                 target_tau=0.01, target_update_step=10):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = env.n_agents
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

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

    # agent interact with the environment to collect experience
    def interact(self):
        pass

    # take one step
    def _take_one_step(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        action = self.exploration_action(state)
        next_state, reward, done, _ = self.env.step(action)

        if done[0]:
            if self.done_penalty is not None:
                reward = np.ones(self.n_agents) * self.done_penalty
            next_state = np.zeros_like(state)
            self.env_state = self.env.reset()
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.env_state = next_state
            self.episode_done = False
        self.n_steps += 1
        self.memory.push(state, action, reward, next_state, done)

    # take n_agents steps
    def _take_n_steps(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        # take n_agents steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = next_state
            actions.append(action)
            if done[0] and self.done_penalty is not None:
                reward = np.ones(self.n_agents) * self.done_penalty
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done[0]:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done[0]:
            final_value = np.zeros(self.n_agents)
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_value = self.value(final_state, final_action)
        rewards = self._discount_reward(rewards, final_value)
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

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

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state) -> np.ndarray:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_episodes / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.env.n_actions, self.env.n_agents)
        else:
            action = self.action(state)
        return action

    # choose an action based on state for execution
    def action(self, state) -> np.ndarray:
        pass

    # evaluate value for a state-action pair
    def value(self, state, action) -> np.ndarray:
        pass

    # evaluation the learned agent
    def evaluation(self, env: EdgeMultiAgentEnv, eval_episodes=10, seed=0):
        rewards = []
        infos = []
        rewards_cache = []
        rewards_switch = []
        users_info = []
        stats = {}
        stats['cache'] = []
        stats['delay'] = []
        stats['ratio'] = []
        average_delays = []
        # seed_everything(seed)

        for i in range(eval_episodes):
            self.mean_actions_e = copy.deepcopy(self.mean_actions)
            stats['delay'] = []
            rewards_i = []
            rewards_cache_i = []
            rewards_switch_i = []
            state = env.reset()
            action = self.action(state)
            state, reward, done, info = env.step(action)

            done = done[0] if isinstance(done, np.ndarray) else done
            rewards_i.append(reward)
            rewards_cache_i.append(info[0])
            rewards_switch_i.append(info[1])
            stats['delay'].append(env.world.average_delay)
            stats['ratio'].append(env.world.cache_hit_ratio)

            while not done:
                action = self.action(state)
                state, reward, done, info = env.step(action)

                done = done[0] if isinstance(done, np.ndarray) else done
                rewards_i.append(reward)
                rewards_cache_i.append(info[0])
                rewards_switch_i.append(info[1])
                stats['delay'].append(env.world.average_delay)
                stats['ratio'].append(env.world.cache_hit_ratio)

            rewards.append(rewards_i)
            rewards_cache.append(rewards_cache_i)
            rewards_switch.append(rewards_switch_i)
            infos.append(env.state_info())
            users_info.append(env.users_info())
            stats['cache'].append(env.world.cache_stat)

        return rewards, infos, rewards_cache, rewards_switch, users_info, stats
