import sys
import os
from typing import Tuple
import gym
from gym import spaces
import math
import numpy as np
import torch as th
from Core import EdgeAgent, EdgeWorld, User, Model
from utils.distribution import generate_requests
from utils.tool import states2array, cache2str

from numpy import ndarray
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EdgeMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, n_agents=100, n_users=30, agent_capacity=100, agent_view=1, n_requests=1000, max_steps=10000,
                 temperature=0.1, eta=1, beta=1, device='cpu'):

        self.world = make_world(n_agents=n_agents, agent_view=agent_view, agent_capacity=agent_capacity,
                                n_users=n_users, n_requests=n_requests, max_steps=max_steps)
        self.agents = self.world.policy_agents
        # number of controllable agents
        self.n_agents = len(self.world.policy_agents)
        assert self.n_agents == len(self.world.agents)
        # ndarray of legal actions represented by integers
        self.n_actions = self.world.n_caches
        self.num2action = self.world.number2cache
        self.n_models = self.world.n_models
        self.n_users = self.world.n_users
        self.temperature = temperature
        self.eta = eta  # weight factor for switch cost
        self.beta = beta  # weight factor for accuracy

        # configure spaces
        # no implementation here
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.MultiBinary(self.n_agents * self.n_models * 2)

    def reset(self, seed=None, return_info=False, options=None):
        # reset world, init agent state and global state
        reset_world(self.world)
        self.agents = self.world.policy_agents

        # return the new global state
        return self._get_state()

    def step(self, action_n):
        reward_n = np.zeros(self.n_agents)
        done_n = np.zeros(self.n_agents)

        # set action for each agent
        for i, agent in enumerate(self.agents):
            set_action(action_n[i], agent, self.world)

        # update the agent's new state and global state
        self.world.step(self.beta)

        rewards_c, rewards_s = self.get_rewards_resp()

        for i, agent in enumerate(self.agents):
            reward_n[i] = self._get_reward(agent)
            done_n[i] = self._get_done(agent)

        state = self._get_state()
        info = [rewards_c, rewards_s]

        return state, reward_n, done_n, info

    # return a formatted string of the global states
    def state_info(self) -> str:
        res = f"{'AGENT ID':>4}  {'CACHE':>{2 * self.n_models - 1}}  {'REQUEST':>{2 * self.n_models - 1}}"
        for i in range(self.n_agents):
            cache = self.world.global_state[i].cache
            cache_str = ','.join(str(j) for j in cache)
            request = self.world.global_state[i].request
            request_str = ','.join(str(j) for j in request)
            res += f"\n{i:8d}  {cache_str}  {request_str}"

        return res

    # return the rewards of cache hit and switch cost of each agent respectively
    def get_rewards_resp(self) -> Tuple[ndarray, ndarray]:
        cache = np.zeros(self.n_agents)
        switch_cost = np.zeros(self.n_agents)

        for i, agent in enumerate(self.agents):
            cache[i] = agent.reward_c
            switch_cost[i] = agent.reward_s

        return cache, switch_cost

    # get reward for a particular agent
    def _get_reward(self, agent) -> float:
        return agent.reward_c - self.eta * agent.reward_s

    # get global state of the world
    def _get_state(self) -> np.ndarray:
        return states2array(self.world.global_state)

    # get observation for a particular agent
    def get_obs(self, agent_id) -> list:
        # return the ids of neighbour agents
        agent = self.agents[agent_id]
        ids = []
        for i, neighbor in enumerate(agent.neighbor_mask):
            if neighbor == 1:
                ids.append(i)
        return ids

    # get dones for a particular agent
    def _get_done(self, agent) -> bool:
        if self.world.n_steps >= self.world.max_steps:
            return True
        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


# helper functions
def make_world(shape_size=np.array([1000, 1000]), n_agents=100, agent_view=1, agent_capacity=20, n_users=10,
               n_requests=1000, max_steps=10000) -> EdgeWorld:
    assert agent_view < n_agents, "agent view must be smaller than the number of agents"
    world = EdgeWorld()
    world.shape_size = shape_size
    world.agent_view_sight = agent_view
    world.n_agents = n_agents
    world.agent_storage = agent_capacity
    world.agents = [EdgeAgent(capacity=world.agent_storage, view_sight=world.agent_view_sight, id=i)
                    for i in range(n_agents)]
    for i, agent in enumerate(world.agents):
        agent.state = world.global_state[i]
        agent.location = np.random.random(2) * world.shape_size
    for agent in world.agents:
        calc_mask(agent, world)

    world.n_models = 10
    world.models = models
    for i, model_type in enumerate(models):
        world.model_sizes.append([])
        for model in model_type:
            world.model_sizes[i].append(model.size)

    world.n_users = n_users
    world.users = [User(user_id=i, loc=np.random.random(2) * world.shape_size, models_num=world.n_models)
                   for i in range(n_users)]
    calc_trans_rate(world)

    world.n_requests = n_requests
    # random order of model types
    world.request_popularity = np.array([np.random.permutation(world.n_models) for _ in range(n_users)])
    world.requests_next = generate_requests(num_users=n_users, num_types=world.n_models,
                                            num_requests=n_requests, orders=world.request_popularity)

    world.max_steps = max_steps

    calc_valid_caches(world)
    world.n_caches = len(world.number2cache)

    # make initial conditions
    reset_world(world)

    return world


def reset_world(world: EdgeWorld):
    world.n_steps = 0
    for s in world.global_state:
        s.cache = world.number2cache[np.random.choice(world.n_caches)]
        s.popularity = np.zeros(world.n_models)


# set env action for a particular agent
def set_action(action: int, agent: EdgeAgent, world: EdgeWorld):
    action = world.number2cache[action]
    agent.action.a = action


# calculate the neighbor mask of agent
def calc_mask(agent: EdgeAgent, world: EdgeWorld):
    # init mask with all zeros
    agent.neighbor_mask = np.zeros(world.n_agents, dtype=np.int8)

    if agent.view_sight == -1:
        # fully observed
        agent.neighbor_mask += 1
    elif agent.view_sight == 0:
        # observe itself
        agent.neighbor_mask[agent.id] = 1
    elif agent.view_sight > 0:
        dis_id = {}
        for a in world.agents:
            dis = np.linalg.norm(a.location - agent.location)
            dis_id[dis] = a.id
        sorted_dict = dict(sorted(dis_id.items()))
        ids = list(sorted_dict.values())[:agent.view_sight + 1]
        agent.neighbor_mask[ids] = 1


# calculate all valid caches that will not exceed storage space of the edge and set the mapping between number and cache
def calc_valid_caches(world: EdgeWorld):
    zeros = np.zeros(world.n_models)
    caches_list = []
    subset_sum(world.model_sizes, world.agent_storage, zeros, world.n_models, caches_list)
    caches_list = np.array(caches_list)
    for i, cache in enumerate(caches_list):
        world.number2cache[i] = cache
        cache_str = cache2str(cache)
        world.cache2number[cache_str] = i


# def legal_actions(world: EdgeWorld) -> th.Tensor:
#     actions = th.zeros(2 ** world.n_contents, dtype=th.bool)
#
#     for i in world.valid_model_indices:
#         action_index = binary2int(i)
#         actions[action_index] = True
#
#     return actions


# return all the possible sublists of numbers using one-hot encoding where the sum <= limit
def subset_sum(numbers, limit, partial, original_len, res=[]):
    res.append(partial)
    if limit == 0:
        return

    for i, nums in enumerate(numbers):
        remaining = [row[:] for row in numbers[i + 1:]]
        front = original_len - len(numbers)
        for j, n in enumerate(nums):
            if limit - n < 0:
                continue
            var_partial = partial.copy()
            var_partial[i + front] = j + 1
            subset_sum(remaining, limit - n, var_partial, original_len, res)


# calculate the transmission rates in the system in Mbps
def calc_trans_rate(world: EdgeWorld):
    bandwidth = 1  # MHz
    wave_length = 0.1  # m
    noise_power = math.pow(10, -17.4) * 1e-3  # -174dBm
    send_power = 0.2  # W

    channel_gain_const = (wave_length / (4 * math.pi)) ** 2
    for i, user in enumerate(world.users):
        for j, edge in enumerate(world.agents):
            x = (user.loc[0] - edge.location[0]) ** 2 + (user.loc[1] - edge.location[1]) ** 2
            channel_gain = channel_gain_const / x
            world.trans_rates_ie[i][j] = bandwidth * math.log2(1 + send_power * channel_gain / noise_power)

    for i in range(world.n_agents):
        world.trans_rates_ec[i] = 150  # Mbps


# multi-agent environment
def action_values_softmax(action_values: th.Tensor) -> th.Tensor:
    action_values = th.softmax(action_values, dim=1)
    action_values = th.clip(action_values, min=1e-10, max=1 - 1e-10)

    return action_values


# model list
models = [
    [
        Model(name='AlexNet', type_id=0, accuracy=56.522, model_size=233.1, time_c=0.25, time_e=10.1, input_size=25.17)
    ],
    [
        Model(name='VGG-11', type_id=1, accuracy=69.02, model_size=506.8, time_c=0.24, time_e=10.36, input_size=25.17),
        Model(name='VGG-13', type_id=1, accuracy=69.928, model_size=507.5, time_c=0.28, time_e=10.51, input_size=25.17),
        Model(name='VGG-16', type_id=1, accuracy=71.592, model_size=527.8, time_c=0.28, time_e=11.29, input_size=25.17),
        Model(name='VGG-19', type_id=1, accuracy=72.376, model_size=548.1, time_c=0.32, time_e=9.59, input_size=25.17),
        Model(name='VGG-11_BN', type_id=1, accuracy=70.37, model_size=506.9, time_c=0.27, time_e=13.17,
              input_size=25.17),
        Model(name='VGG-13_BN', type_id=1, accuracy=71.586, model_size=507.6, time_c=0.27, time_e=14.82,
              input_size=25.17),
        Model(name='VGG-16_BN', type_id=1, accuracy=73.36, model_size=527.9, time_c=0.29, time_e=12.76,
              input_size=25.17),
        Model(name='VGG-19_BN', type_id=1, accuracy=74.218, model_size=548.1, time_c=0.34, time_e=14.97,
              input_size=25.17)
    ],
    [
        Model(name='ResNet-18', type_id=2, accuracy=69.758, model_size=44.7, time_c=0.38, time_e=6.07,
              input_size=25.17),
        Model(name='ResNet-34', type_id=2, accuracy=73.314, model_size=83.3, time_c=0.56, time_e=9.08,
              input_size=25.17),
        Model(name='ResNet-50', type_id=2, accuracy=76.13, model_size=97.8, time_c=0.51, time_e=9.52,
              input_size=25.17),
        Model(name='ResNet-101', type_id=2, accuracy=77.374, model_size=170.5, time_c=0.78, time_e=14.9,
              input_size=25.17),
        Model(name='ResNet-152', type_id=2, accuracy=78.312, model_size=230.4, time_c=1.03, time_e=20.72,
              input_size=25.17)
    ],
    [
        Model(name='SqueezeNet-v1.0', type_id=3, accuracy=58.092, model_size=4.8, time_c=1.98, time_e=2.04,
              input_size=25.17),
        Model(name='SqueezeNet-v1.1', type_id=3, accuracy=58.178, model_size=4.7, time_c=2.27, time_e=8.56,
              input_size=25.17)
    ],
    [
        Model(name='DenseNet-121', type_id=4, accuracy=74.434, model_size=30.8, time_c=1.93, time_e=20.18,
              input_size=25.17),
        Model(name='DenseNet-169', type_id=4, accuracy=75.6, model_size=54.7, time_c=2.90, time_e=29.89,
              input_size=25.17),
        Model(name='DenseNet-201', type_id=4, accuracy=76.896, model_size=77.4, time_c=3.98, time_e=41.25,
              input_size=25.17),
        Model(name='DenseNet-161', type_id=4, accuracy=77.138, model_size=110.4, time_c=2.19, time_e=33.28,
              input_size=25.17)
    ]
]
