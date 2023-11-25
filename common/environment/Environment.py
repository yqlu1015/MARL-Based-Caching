import sys
import os
from typing import Tuple
import gym
from gym import spaces
import math
import numpy as np
import torch as th
from common.environment.Core import EdgeState, EdgeAgent, EdgeWorld, User, Model
from common.environment.utils.distribution import generate_requests
from common.environment.utils.tool import states2array, normalize

from numpy import ndarray

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EdgeMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, n_agents=100, n_users=30, agent_capacity=100, agent_view=1, n_requests=1000, max_steps=10000,
                 temperature=0.1, eta=1, beta=1, seed=2023, zipf_param=0.8, add_ppl=True):
        np.random.seed(seed)
        self.world = make_world(n_agents=n_agents, agent_view=agent_view, agent_capacity=agent_capacity,
                                n_users=n_users, n_requests=n_requests, max_steps=max_steps, zipf_param=zipf_param)
        self.agents = self.world.policy_agents
        self.agents_loc = np.array([agent.location for agent in self.world.agents])
        self.users_loc = np.array([user.loc for user in self.world.users])
        # number of controllable agents
        self.n_agents = len(self.world.policy_agents)
        assert self.n_agents == len(self.world.agents)
        # ndarray of legal actions represented by integers
        self.n_actions = self.world.n_caches
        self.num2action = self.world.number2cache
        self.n_models = self.world.n_model_types
        self.n_users = self.world.n_users
        self.temperature = temperature
        self.eta = eta  # weight factor for switch cost
        self.beta = beta  # weight factor for accuracy
        self.add_ppl = add_ppl  # whether adding model popularity to the state or not

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

        rewards_c, rewards_s = self._get_rewards_resp()

        for i, agent in enumerate(self.agents):
            reward_n[i] = self._get_reward(agent)
            done_n[i] = self._get_done(agent)

        state = self._get_state()
        # get global state which is used for critic
        global_state = self.get_global_state()
        info = [rewards_c, rewards_s, global_state]

        return state, reward_n, done_n, info

    # return the rewards of avg_qoe and switch cost of each agent respectively
    def _get_rewards_resp(self) -> Tuple[ndarray, ndarray]:
        qoe = np.zeros(self.n_agents)
        switch_cost = np.zeros(self.n_agents)

        for i, agent in enumerate(self.agents):
            qoe[i] = agent.avg_qoe
            switch_cost[i] = agent.switch

        return qoe, switch_cost

    # get reward for a particular agent
    def _get_reward(self, agent) -> float:
        return self.eta * self.world.avg_qoe - self.world.switch_sum
        # return agent.avg_qoe - agent.switch
        # return (agent.avg_delay_old - agent.avg_qoe) + \
        #        self.eta * (self.world.average_delay_old - self.world.average_delay)

    # get observations of the world
    def _get_state(self) -> np.ndarray:
        # if not self.add_ppl:
        #     states = np.zeros(self.n_actions * self.n_agents)
        #     for i, state in enumerate(self.world.observations):
        #         cache_idx = self.world.cache2number[state.cache.tobytes()]
        #         states[i * self.n_actions + cache_idx] = 1
        # else:
        #     # [cache0, ppl0, cache1, ppl1, ...]
        #     states = np.zeros((self.n_actions + self.n_models) * self.n_agents)
        #     for i, state in enumerate(self.world.observations):
        #         cache_idx = self.world.cache2number[state.cache.tobytes()]
        #         states[i * (self.n_actions + self.n_models) + cache_idx] = 1
        #         ppl = state.popularity
        #         start_idx = i * (self.n_actions + self.n_models) + self.n_actions
        #         end_idx = (i+1) * (self.n_actions + self.n_models)
        #         states[start_idx: end_idx] = ppl
        # return states
        return states2array(self.world.observations, self.add_ppl)

    # get neighboring agents for a particular agent
    def get_obs(self, agent_id) -> list:
        # return the ids of neighbour agents
        agent = self.agents[agent_id]
        ids = []
        for i, neighbor in enumerate(agent.neighbor_mask):
            if neighbor == 1 and i != agent_id:
                ids.append(i)
        return ids

    # global state: all requests, last caching decisions and agent popularity
    def get_global_state(self) -> np.ndarray:
        requests_num = np.ravel(self.world.requests_next) / np.max(self.world.requests_next)
        return np.ravel([np.concatenate((requests_num, obs.popularity, obs.cache)) for obs in self.world.observations])
        # return requests_num
        # states = self.world.observations
        # return np.ravel([state.popularity for state in states])

    # get dones for a particular agent
    def _get_done(self, agent) -> bool:
        if self.world.n_steps >= self.world.max_steps:
            self.world.n_steps = 0
            return True
        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    # return a formatted string of the global states
    def state_info(self) -> str:
        res = f"{'AGENT':>4}  {'CACHE':>{2 * self.n_models - 1}}  {'POPULARITY':>{5 * self.n_models - 1}}  " \
              f"{'Average QoE':>4}"
        for i in range(self.n_agents):
            cache = self.world.observations[i].cache
            cache_str = ','.join(str(j) for j in cache)
            popularity = self.world.observations[i].popularity
            popularity_str = ','.join("{:.2f}".format(j) for j in popularity)
            qoe = self.world.agents[i].avg_qoe
            res += f"\n{i:8d}  {cache_str}  {popularity_str} {qoe:.2f}"

        return res

    # return users' information
    def users_info(self) -> str:
        res = f"{'USER':>4} {'Target':>{2 * self.n_models - 1}} {'Accuracy (%)':>{5 * self.n_models - 1}} " \
              f"{'Transmission Delay (s)':>{5 * self.n_models - 1}} {'Inference Time (s)':>{5 * self.n_models - 1}}"
        for i in range(self.n_users):
            target = self.world.users[i].target
            target_str = ','.join(str(t) for t in target)
            acc = self.world.users[i].accuracy
            acc_str = ','.join("{:.1f}".format(a) for a in acc)
            trans = self.world.users[i].trans_delay
            trans_str = ','.join("{:.1f}".format(t) for t in trans)
            inf = self.world.users[i].inf_delay
            inf_str = ','.join("{:.1f}".format(i) for i in inf)
            res += f"\n{i:4d}  {target_str}  {acc_str}  {trans_str}  {inf_str}"

        return res

    # multi-agent environment
    def action_values_softmax(self, action_values: th.Tensor) -> th.Tensor:
        action_values = th.softmax(action_values / self.temperature, dim=1)
        action_values = th.clip(action_values, min=1e-10, max=1 - 1e-10)

        return action_values


# helper functions
def make_world(shape_size=np.array([1000, 1000], dtype='float'), n_agents=5, agent_view=1, agent_capacity=20,
               n_users=15,
               n_requests=1000, max_steps=10000, zipf_param=0.8, user_density=3) -> EdgeWorld:
    world = EdgeWorld(n_agents=n_agents, n_users=n_users)
    world.shape_size = shape_size

    world.agent_view_sight = min(agent_view, n_agents - 1)
    world.agent_storage = agent_capacity
    world.agents = [EdgeAgent(capacity=world.agent_storage, view_sight=world.agent_view_sight, id=i)
                    for i in range(n_agents)]
    # agent_locs = [[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]]
    # # generate locations within circles centering at agent_locs
    # rads = [[0, .5], [.5, 1], [1, 1.5], [1.5, 2], [0, 2]]
    # user_locs = []
    # for i in range(n_users):
    #     agent_idx = i // user_density
    #     limit = rads[agent_idx]
    #     theta = (np.random.rand() * (limit[1] - limit[0]) + limit[0]) * np.pi
    #     loc = [0.1 * np.cos(theta), 0.1 * np.sin(theta)] + agent_locs[agent_idx]
    #     user_locs.append(loc)

    for i, agent in enumerate(world.agents):
        agent.state = world.observations[i]
        agent.location = np.random.random(2) * world.shape_size
        # agent.location = world.shape_size * agent_locs[i]
    for agent in world.agents:
        calc_mask(agent, world)

    world.n_model_types = len(models)
    world.models = [row[::-1] for row in models]
    for i, model_type in enumerate(world.models):
        world.n_models += len(model_type)
        world.model_input_sizes.append(model_type[0].input_size)
        world.model_sizes.append([])
        for model in model_type:
            world.model_sizes[i].append(model.size)
        world.model_sizes[i] = normalize(world.model_sizes[i])
    world.global_popularity = np.zeros(world.n_model_types)

    world.users = [User(user_id=i, loc=np.random.random(2) * world.shape_size, models_num=world.n_model_types)
                   for i in range(n_users)]
    # world.users = [User(user_id=i, loc=world.shape_size * user_locs[i], models_num=world.n_model_types)
    #                for i in range(n_users)]
    calc_trans_rate(world)

    world.n_requests = n_requests
    # random order of model types
    world.request_popularity = np.tile(np.arange(world.n_model_types), (n_users, 1))
    world.zipf_param = zipf_param

    world.max_steps = max_steps

    calc_valid_caches(world)
    world.n_caches = len(world.number2cache)

    # make initial conditions
    reset_world(world)

    return world


def reset_world(world: EdgeWorld):
    world.n_steps = 0
    for s in world.observations:
        s.cache = world.number2cache[np.random.choice(world.n_caches)]
        s.popularity = np.zeros(world.n_model_types)
    world.requests_next = generate_requests(num_users=world.n_users, num_types=world.n_model_types,
                                            num_low=int(world.n_requests * 0.8), num_high=int(world.n_requests * 1.2),
                                            orders=world.request_popularity, zipf_param=world.zipf_param)
    world.global_popularity = np.sum(world.requests_next, axis=0) / np.sum(world.requests_next)


# set env action for a particular agent
def set_action(action: int, agent: EdgeAgent, world: EdgeWorld):
    action = world.number2cache[action]
    agent.action.a = action


# calculate the neighbor mask of agent
def calc_mask(agent: EdgeAgent, world: EdgeWorld):
    # init mask with all zeros
    agent.neighbor_mask = np.zeros(world.n_agents, dtype=np.int8)

    if agent.view_sight >= world.n_agents - 1 or agent.view_sight < 0:
        # observe
        agent.neighbor_mask = np.ones(world.n_agents, dtype=np.int8)
        agent.neighbor_mask[agent.id] = 0
    elif agent.view_sight > 0:
        dis_id = {}
        for a in world.agents:
            if a == agent:
                continue
            dis = np.linalg.norm(a.location - agent.location)
            dis_id[dis] = a.id
        sorted_dict = dict(sorted(dis_id.items()))
        ids = list(sorted_dict.values())[:agent.view_sight]
        agent.neighbor_mask[ids] = 1



# calculate all valid caches that will not exceed storage space of the edge and set the mapping between number and cache
def calc_valid_caches(world: EdgeWorld):
    zeros = np.zeros(world.n_model_types, dtype=np.int32)
    caches_list = []
    subset_sum(world.model_sizes, world.agent_storage, zeros, world.n_model_types, caches_list)
    caches_list = np.array(caches_list)
    for i, cache in enumerate(caches_list):
        world.number2cache.append(cache)
        world.cache2number[cache.tobytes()] = i


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
    bandwidth = 10  # MHz from UAV
    wave_length = 0.1  # m
    noise_power = math.pow(10, -17.4) * 1e3 * bandwidth  # -174dBm/Hz from gao
    send_power = 0.2  # W from UAV

    channel_gain_const = (wave_length / (4 * math.pi)) ** 2
    for i, user in enumerate(world.users):
        for j, edge in enumerate(world.agents):
            x = (user.loc[0] - edge.location[0]) ** 2 + (user.loc[1] - edge.location[1]) ** 2
            channel_gain = channel_gain_const / x
            world.trans_rates_ie[i][j] = bandwidth * math.log2(1 + send_power * channel_gain / noise_power)

    for i in range(world.n_agents):
        world.trans_rates_ec[i] = 10  # Mbps from UAV


# model list
models = [
    [
        Model(name='ResNet-50', type='Type 1', accuracy=76.13, model_size=97.8, time_c=0.51, time_e=9.52,
              input_size=1.204),
        Model(name='ResNet-101', type='Type 1', accuracy=77.374, model_size=170.5, time_c=0.78, time_e=14.9,
              input_size=1.204),
        Model(name='ResNet-152', type='Type 1', accuracy=78.312, model_size=230.4, time_c=1.03, time_e=20.72,
              input_size=1.204)
    ],
    [
        Model(name='ResNet-18', type='Type 2', accuracy=69.758, model_size=44.7, time_c=0.38, time_e=6.07,
              input_size=1.204),
        Model(name='ResNet-34', type='Type 2', accuracy=73.314, model_size=83.3, time_c=0.56, time_e=9.08,
              input_size=1.204),
    ],
    [
        Model(name='ResNeXt-50(32x4d)', type='Type 3', accuracy=77.618, model_size=95.8, time_c=0.67, time_e=14.76,
              input_size=1.204),
        Model(name='ResNeXt-101(32x8d)', type='Type 3', accuracy=79.312, model_size=339.6, time_c=0.67, time_e=35.69,
              input_size=1.204),
    ],
    [
        Model(name='VGG-11', type='Type 4', accuracy=69.02, model_size=506.8, time_c=0.24, time_e=10.36,
              input_size=1.204),
        Model(name='VGG-13', type='Type 4', accuracy=69.928, model_size=507.5, time_c=0.28, time_e=10.51,
              input_size=1.204),
        Model(name='VGG-16', type='Type 4', accuracy=71.592, model_size=507.8, time_c=0.28, time_e=11.29,
              input_size=1.204),
        Model(name='VGG-19', type='Type 4', accuracy=72.376, model_size=508.1, time_c=0.32, time_e=9.59,
              input_size=1.204)
    ],
    [
        Model(name='VGG-11_BN', type='Type 5', accuracy=70.37, model_size=506.9, time_c=0.27, time_e=13.17,
              input_size=1.204),
        Model(name='VGG-13_BN', type='Type 5', accuracy=71.586, model_size=507.6, time_c=0.27, time_e=14.82,
              input_size=1.204),
        Model(name='VGG-16_BN', type='Type 5', accuracy=73.36, model_size=527.9, time_c=0.29, time_e=12.76,
              input_size=1.204),
        Model(name='VGG-19_BN', type='Type 5', accuracy=74.218, model_size=548.1, time_c=0.34, time_e=14.97,
              input_size=1.204)
    ],
    [
        Model(name='DenseNet-121', type='Type 6', accuracy=74.434, model_size=30.8, time_c=1.93, time_e=20.18,
              input_size=1.204),
        Model(name='DenseNet-169', type='Type 6', accuracy=75.6, model_size=54.7, time_c=2.90, time_e=29.89,
              input_size=1.204)
    ],
    [
        Model(name='DenseNet-201', type='Type 7', accuracy=76.896, model_size=77.4, time_c=3.98, time_e=41.25,
              input_size=1.204),
        Model(name='DenseNet-161', type='Type 7', accuracy=77.138, model_size=110.4, time_c=2.19, time_e=33.28,
              input_size=1.204)
    ],
    [
        Model(name='Inception-v1', type='Type 8', accuracy=69.778, model_size=49.7, time_c=2.91, time_e=17.77,
              input_size=2.146),
        Model(name='Inception-v3', type='Type 8', accuracy=77.294, model_size=103.9, time_c=4.21, time_e=86.44,
              input_size=2.146)
    ],
    [
        Model(name='SqueezeNet-v1.0', type='Type 9', accuracy=58.092, model_size=4.8, time_c=1.98, time_e=2.04,
              input_size=1.237),
        Model(name='SqueezeNet-v1.1', type='Type 9', accuracy=58.178, model_size=4.7, time_c=2.27, time_e=8.56,
              input_size=1.237)
    ],
    [
        Model(name='MNASNet-0.5', type='Type 10', accuracy=67.734, model_size=86., time_c=2.5, time_e=18.1,
              input_size=1.204),
        Model(name='MNASNet-1.0', type='Type 10', accuracy=73.456, model_size=169., time_c=3.49, time_e=28.1,
              input_size=1.204)
    ]
    # [
    #     Model(name='ResNet-50', type='Type 11', accuracy=76.13, model_size=97.8, time_c=0.51, time_e=9.52,
    #           input_size=1.204),
    #     Model(name='ResNet-101', type='Type 11', accuracy=77.374, model_size=170.5, time_c=0.78, time_e=14.9,
    #           input_size=1.204),
    #     Model(name='ResNet-152', type='Type 11', accuracy=78.312, model_size=230.4, time_c=1.03, time_e=20.72,
    #           input_size=1.204)
    # ],
    # [
    #     Model(name='ResNet-18', type='Type 12', accuracy=69.758, model_size=44.7, time_c=0.38, time_e=6.07,
    #           input_size=1.204),
    #     Model(name='ResNet-34', type='Type 12', accuracy=73.314, model_size=83.3, time_c=0.56, time_e=9.08,
    #           input_size=1.204),
    # ],
    # [
    #     Model(name='ResNeXt-50(32x4d)', type='Type 13', accuracy=77.618, model_size=95.8, time_c=0.67, time_e=14.76,
    #           input_size=1.204),
    #     Model(name='ResNeXt-101(32x8d)', type='Type 13', accuracy=79.312, model_size=339.6, time_c=0.67, time_e=35.69,
    #           input_size=1.204),
    # ],
    # [
    #     Model(name='VGG-11', type='Type 14', accuracy=69.02, model_size=506.8, time_c=0.24, time_e=10.36,
    #           input_size=1.204),
    #     Model(name='VGG-13', type='Type 14', accuracy=69.928, model_size=507.5, time_c=0.28, time_e=10.51,
    #           input_size=1.204),
    #     Model(name='VGG-16', type='Type 14', accuracy=71.592, model_size=507.8, time_c=0.28, time_e=11.29,
    #           input_size=1.204),
    #     Model(name='VGG-19', type='Type 14', accuracy=72.376, model_size=508.1, time_c=0.32, time_e=9.59,
    #           input_size=1.204)
    # ],
    # [
    #     Model(name='VGG-11_BN', type='Type 15', accuracy=70.37, model_size=506.9, time_c=0.27, time_e=13.17,
    #           input_size=1.204),
    #     Model(name='VGG-13_BN', type='Type 15', accuracy=71.586, model_size=507.6, time_c=0.27, time_e=14.82,
    #           input_size=1.204),
    #     Model(name='VGG-16_BN', type='Type 15', accuracy=73.36, model_size=527.9, time_c=0.29, time_e=12.76,
    #           input_size=1.204),
    #     Model(name='VGG-19_BN', type='Type 15', accuracy=74.218, model_size=548.1, time_c=0.34, time_e=14.97,
    #           input_size=1.204)
    # ],
    # [
    #     Model(name='DenseNet-121', type='Type 16', accuracy=74.434, model_size=30.8, time_c=1.93, time_e=20.18,
    #           input_size=1.204),
    #     Model(name='DenseNet-169', type='Type 16', accuracy=75.6, model_size=54.7, time_c=2.90, time_e=29.89,
    #           input_size=1.204)
    # ],
    # [
    #     Model(name='DenseNet-201', type='Type 17', accuracy=76.896, model_size=77.4, time_c=3.98, time_e=41.25,
    #           input_size=1.204),
    #     Model(name='DenseNet-161', type='Type 17', accuracy=77.138, model_size=110.4, time_c=2.19, time_e=33.28,
    #           input_size=1.204)
    # ],
    # [
    #     Model(name='Inception-v1', type='Type 18', accuracy=69.778, model_size=49.7, time_c=2.91, time_e=17.77,
    #           input_size=2.146),
    #     Model(name='Inception-v3', type='Type 18', accuracy=77.294, model_size=103.9, time_c=4.21, time_e=86.44,
    #           input_size=2.146)
    # ],
    # [
    #     Model(name='SqueezeNet-v1.0', type='Type 19', accuracy=58.092, model_size=4.8, time_c=1.98, time_e=2.04,
    #           input_size=1.237),
    #     Model(name='SqueezeNet-v1.1', type='Type 19', accuracy=58.178, model_size=4.7, time_c=2.27, time_e=8.56,
    #           input_size=1.237)
    # ],
    # [
    #     Model(name='MNASNet-0.5', type='Type 20', accuracy=67.734, model_size=86., time_c=2.5, time_e=18.1,
    #           input_size=1.204),
    #     Model(name='MNASNet-1.0', type='Type 20', accuracy=73.456, model_size=169., time_c=3.49, time_e=28.1,
    #           input_size=1.204)
    # ]
]
