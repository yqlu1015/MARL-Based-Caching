import sys
import os
from typing import Tuple

from numpy import ndarray
from torch import Tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gym
from gym import spaces
import random
import numpy as np
import torch as th
from Core import EdgeAgent, EdgeWorld
from utils.distribution import generate_content_sizes, generate_requests
from utils.tool import states2array, binary2int, int2binary


# multi-agent environment
class EdgeMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, n_agents=100, n_contents=50, agent_capacity=100, agent_view=1, n_requests=1000, max_steps=10000,
                 temperature=0.1, replacement_factor=1, device='cpu'):

        self.world = make_world(n_agents=n_agents, agent_view=agent_view, agent_capacity=agent_capacity,
                                n_contents=n_contents, n_requests=n_requests, max_steps=max_steps)
        self.agents = self.world.policy_agents
        # number of controllable agents
        self.n = len(self.world.policy_agents)
        assert self.n == len(self.world.agents)
        # ndarray of legal actions represented by integers
        self.legal_actions = self.world.valid_content_indices
        self.legal_actions_mask = self.world.legal_actions_mask.to(device)
        self.n_contents = self.world.n_contents
        self.n_actions = 2 ** self.world.n_contents
        self.temperature = temperature
        self.replacement_factor = replacement_factor

        # configure spaces
        # no implementation here
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.MultiBinary(self.n * self.n_contents * 2)

        # self.action_space.append(spaces.Discrete(self.world.n_contents))
        # # observation space, called self-defined scenario.observation
        # # define the size of the observation here
        # # use the global state + mask
        # self.observation_space.append(spaces.MultiBinary(
        #   4 * self.world.agent_view_sight))

    def reset(self, seed=None, return_info=False, options=None):
        # reset world, init agent state and global state
        reset_world(self.world)
        self.agents = self.world.policy_agents

        # return the new global state
        return self._get_state()

    def step(self, action_n):
        reward_n = np.zeros(self.n)
        done_n = np.zeros(self.n)

        # set action for each agent
        for i, agent in enumerate(self.agents):
            set_action(action_n[i], agent, self.world)

        # update the agent's new state and global state
        request_nums = self.world.step()

        rewards_c, rewards_s = self.get_rewards_resp(request_nums)

        for i, agent in enumerate(self.agents):
            reward_n[i] = self._get_reward(agent)
            done_n[i] = self._get_done(agent)
            update_state_cache(agent, self.world)

        state = self._get_state()
        info = [rewards_c, rewards_s]

        return state, reward_n, done_n, info

    # return the masked action values
    # assume all agents are homogeneous
    def action_values_mask(self, action_values: th.Tensor) -> th.Tensor:
        mask_value = th.finfo(action_values.dtype).min
        action_values.masked_fill_(~self.legal_actions_mask, mask_value)

        return action_values

    def action_values_softmax(self, action_values: th.Tensor) -> th.Tensor:
        mask_value = th.finfo(action_values.dtype).min
        action_values.masked_fill_(~self.legal_actions_mask, mask_value)
        action_values = th.softmax(action_values, dim=1)
        action_values = th.clip(action_values, min=1e-10, max=1 - 1e-10)

        return action_values

    # return a formatted string of the global states
    def state_info(self) -> str:
        res = f"{'AGENT ID':>4}  {'CACHE':>{2 * self.n_contents - 1}}  {'REQUEST':>{2 * self.n_contents - 1}}"
        for i in range(self.n):
            cache = self.world.global_state[i].cache
            cache_str = ','.join(str(j) for j in cache)
            request = self.world.global_state[i].request
            request_str = ','.join(str(j) for j in request)
            res += f"\n{i:8d}  {cache_str}  {request_str}"

        return res

    # return the rewards of cache hit and switch cost of each agent respectively
    # receive the content-size-based weighted number of requests as a parameter
    def get_rewards_resp(self, request_nums) -> Tuple[ndarray, ndarray]:
        cache = np.zeros(self.n)
        switch_cost = np.zeros(self.n)

        for i, agent in enumerate(self.agents):
            # cache hit rate
            r_c = sum(self.world.content_sizes * agent.action.a * agent.state.request) / request_nums
            cache[i] = r_c

            # switch cost
            diff = agent.action.a - agent.state.cache
            switch = np.where(diff < 0, 0, diff)  # indices of new requested contents
            r_s = sum(self.world.content_sizes * switch)
            switch_cost[i] = r_s

        return cache, switch_cost

    # get global state of the world
    def _get_state(self) -> np.ndarray:
        return states2array(self.world.global_state)

    # get observation for a particular agent
    def get_obs(self, agent_id) -> list:
        # return the ids of neighbour agents
        agent = self.world.agents[agent_id]
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

    # get reward for a particular agent
    def _get_reward(self, agent) -> float:
        # cache hit
        epsilon = 1  # incentive price
        r_c = epsilon * sum(self.world.content_sizes * agent.action.a * agent.state.request)

        # switch cost
        diff = agent.action.a - agent.state.cache
        switch = np.where(diff < 0, 0, diff)  # indices of new requested contents
        mu = self.replacement_factor  # weight factor of the replacement cost
        r_s = mu * sum(self.world.content_sizes * switch)

        return r_c - r_s

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


# helper functions
def make_world(shape_size=[10, 10], n_agents=100, agent_view=1, agent_capacity=20,
               n_contents=50, content_size_mean=1, min_content_size=0.5, n_requests=1000, max_steps=10000) -> EdgeWorld:
    assert agent_view < n_agents, "agent view must be smaller than the number of agents"
    world = EdgeWorld()
    world.shape_size = shape_size
    world.agent_view_sight = min(agent_view, n_agents - 1)  # the number of neighbors cannot exceed the number of all
    # other agents
    world.n_agents = n_agents
    world.agents = [EdgeAgent(view_sight=world.agent_view_sight, capacity=agent_capacity, id=i)
                    for i in range(n_agents)]
    for agent in world.agents:
        agent.location = np.random.random(2) * world.shape_size
    for agent in world.agents:
        calc_mask(agent, world)

    world.n_contents = n_contents
    world.content_sizes = generate_content_sizes(n=n_contents, mean=content_size_mean, mini=min_content_size)
    # for agent in world.agents:
    #     agent.state.cache = np.zeros(n_contents, dtype=np.uint8)
    #     agent.state.request = np.zeros(n_contents, dtype=np.uint8)

    world.global_state = np.array([agent.state for agent in world.agents])
    world.requests_next = generate_requests(num_types=n_contents, num_requests=n_requests)
    world.n_requests = n_requests

    world.max_steps = max_steps

    world.valid_content_indices, world.legal_actions_mask = valid_caches(world, world.agents[0])

    # make initial conditions
    reset_world(world)

    return world


def reset_world(world: EdgeWorld):
    world.n_steps = 0

    # init agent state and global state
    for i, agent in enumerate(world.agents):
        agent.id = i
        agent.state.cache = int2binary(np.random.choice(world.valid_content_indices), world.n_contents)
        agent.state.request = np.zeros(world.n_contents, dtype=np.int8)
        world.global_state[i] = agent.state


# set env action for a particular agent
def set_action(action: int, agent: EdgeAgent, world: EdgeWorld):
    action = int2binary(action, world.n_contents)
    agent.action.a = action


# update cached dnns for a particular agent
def update_state_cache(agent: EdgeAgent, world: EdgeWorld):
    agent.state.cache = agent.action.a
    world.global_state[agent.id].cache = agent.state.cache


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


# return a list of all valid caches that will not exceed storage space of the edge and an action mask
def valid_caches(world: EdgeWorld, agent: EdgeAgent) -> Tuple[ndarray, Tensor]:
    res = []
    zeros = np.zeros(world.n_contents, dtype=np.int8)
    actions = th.zeros(2 ** world.n_contents, dtype=th.bool)

    g = subset_sum(world.content_sizes, agent.capacity, zeros)
    for i in g:
        action_index = binary2int(i)
        res.append(action_index)
        actions[action_index] = True

    return np.array(res), actions


def legal_actions(world: EdgeWorld) -> th.Tensor:
    actions = th.zeros(2 ** world.n_contents, dtype=th.bool)

    for i in world.valid_content_indices:
        action_index = binary2int(i)
        actions[action_index] = True

    return actions


# return all the possible sublists of numbers using one-hot encoding where the sum <= limit
def subset_sum(numbers, limit, partial, partial_sum=0, start_index=0):
    if partial_sum > limit:
        return
    else:
        yield partial

    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        var_partial = partial.copy()
        var_partial[i + start_index] = 1
        yield from subset_sum(remaining, limit, var_partial, partial_sum + n, start_index + 1)
