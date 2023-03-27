import numpy as np
import torch as th
from Core import EdgeWorld, EdgeAgent
from utils.distribution import generate_content_sizes, generate_requests
from utils.tool import states2array, binary2int, int2binary

class Scenario():

    def make_world(self, shape_size=[100,100], n_agents=100, agent_view=1, agent_capacity=100, 
            n_contents=50, content_size_mean=10, n_requests=1000, max_steps=10000) -> EdgeWorld:
        assert agent_view < n_agents, "agent view must be smaller than the number of agents"
        world = EdgeWorld()
        world.shape_size = shape_size
        world.agent_view_sight = agent_view 
        world.n_agents = n_agents
        world.agents = [EdgeAgent(capacity=agent_capacity, view_sight=world.agent_view_sight, id=i)
                        for i in range(n_agents)]

        world.n_contents = n_contents
        world.content_sizes = generate_content_sizes(n=n_contents, mean=content_size_mean)

        world.global_state = np.array([agent.state for agent in world.agents])
        world.requests_next = generate_requests(num_types=n_contents, num_requests=n_requests)
        world.n_requests = n_requests

        world.max_steps = max_steps

        world.valid_model_indices = self._valid_caches(world, world.agents[0])
        world.legal_actions_mask = self._legal_actions(world)

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world: EdgeWorld):
        world.n_steps = 0

        # init agent state and global state
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.state.cache = np.random.choice(world.valid_model_indices, 1)
            agent.state.request = np.zeros(world.n_contents, dtype=np.uint8)
            agent.location = np.random.random(2) * world.shape_size 
            world.global_state[i] = agent.state
            self._calc_mask(agent,world)
    
    # calculate the reward of agent after step, mu is the weight factor of switch cost
    def reward(self, agent: EdgeAgent, world: EdgeWorld, mu: float=0.5) -> float:
        # cache hit
        r_c = sum(world.content_sizes * agent.action.a * agent.state.request)

        # switch cost
        diff = agent.action.a-agent.state.cache
        switch = np.where(diff<0,0,diff) # indices of new requested contents
        r_s = mu * sum(world.content_sizes * switch)

        return r_c - r_s

    # return the global state
    def state(self, world: EdgeWorld) -> np.ndarray:
        return states2array(world.global_state)

    def observation(self, agent_id: int, world: EdgeWorld) -> list:
        # return the ids of neighbour agents
        agent = world.agents[agent_id]
        ids = []
        for i, neighbor in enumerate(agent.neighbor_mask):
            if neighbor == 1: 
                ids.append(i)
        return ids

    def done(self, agent: EdgeAgent, world: EdgeWorld) -> bool:
        if world.n_steps >= world.max_steps:
            return True
        return False

    # set env action for a particular agent
    def set_action(self, action: int, agent: EdgeAgent, world: EdgeWorld):
        action = int2binary(action, world.n_contents)
        agent.action.a = action

    # update cached dnns for a particular agent
    def update_state_cache(self, agent: EdgeAgent, world: EdgeWorld):
        agent.state.cache = agent.action.a
        world.global_state[agent.id].cache = agent.state.cache



    # calculate the neighbor mask of agent
    def _calc_mask(self, agent: EdgeAgent, world: EdgeWorld):
        # init mask with all zeros
        agent.neighbor_mask = np.zeros(world.n_agents,dtype=np.uint8)

        if agent.view_sight == -1:
            # fully observed
            agent.neighbor_mask += 1
        elif agent.view_sight == 0:
            # observe itself
            agent.neighbor_mask[agent.id] = 1
        elif agent.view_sight > 0:
            dis_id = {}
            for a in world.agents:
                dis = np.linalg.norm(a.location-agent.location)
                dis_id[dis] = a.id
            sorted_dict = dict(sorted(dis_id.items()))
            ids = list(sorted_dict.values())[1:agent.view_sight+1]
            agent.neighbor_mask[ids] = 1


    # return a list of all valid caches that will not exceed storage space of the edge
    def _valid_caches(self, world: EdgeWorld, agent: EdgeAgent) -> list:
        res = []
        zeros = np.zeros(world.n_contents,dtype=np.uint8)

        g = self.subset_sum(agent.capacity, zeros)
        for i in g:
            res.append(i)

        return res

    # return the masked action values
    # assume all agents are homogeneous
    def legal_action_values(self, world: EdgeWorld, action_values: th.Tensor):
        values = th.ones_like(action_values) * float('-inf')
        zeros = np.zeros(world.n_contents,dtype=np.uint8)

        g = self.subset_sum(world.agents[0].capacity, zeros)
        for i in g:
            action = binary2int(i)
            values[action] = action_values[action]

        return th.nan_to_num(values)

    def _legal_actions(self, world: EdgeWorld):
        actions = []
        zeros = np.zeros(world.n_contents,dtype=np.uint8)

        g = self.subset_sum(world.agents[0].capacity, zeros)
        for i in g:
            action = binary2int(i)
            actions.append(action)

        actions = np.array(actions)
        return actions

    # return all the possible sublists of numbers using one-hot encoding where the sum <= limit
    def subset_sum(numbers, limit, partial, partial_sum=0, start_index=0):
        if partial_sum > limit:
            return
        else:
            yield partial

        for i, n in enumerate(numbers):
            remaining = numbers[i+1:]
            var_partial = partial.copy()
            var_partial[i+start_index] = 1
            yield from self.subset_sum(limit, var_partial, partial_sum + n, start_index + 1)
