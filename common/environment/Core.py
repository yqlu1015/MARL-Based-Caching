import numpy as np
from typing import List

from utils.distribution import generate_requests


class EdgeAgentState(object):
    def __init__(self):
        self.request = []  # numbers of requests of each dnn
        self.cache = []  # one-hot vector of cached dnns


class EdgeAction(object):
    def __init__(self):
        # action - one-hot vector of newly cached dnns
        self.a = []


class EdgeAgent(object):
    def __init__(self, capacity=100, location=np.array([0, 0]), view_sight=10, id=-1):
        self.id = id
        # storage capacity
        self.capacity = capacity
        # geographic location
        self.location = location
        # -1: observe the whole state, 0: itself, n: cloeset n neighbors
        self.view_sight = view_sight
        self.neighbor_mask = None  # the mask for who are neighbors
        # action
        self.action = EdgeAction()
        # state
        self.state = EdgeAgentState()
        # script behavior to execute
        self.action_callback = None


class EdgeWorld(object):
    def __init__(self):
        # shape of the world, i.e. area covered by all edges
        self.shape_size = [0, 0]

        # list of agents (can change at execution-time!)
        self.agents = []
        self.n_agents = 0
        self.agent_view_sight = 1

        # sizes of dnns
        self.content_sizes = []
        self.n_contents = 0

        # state information
        self.global_state = np.array([EdgeAgentState() for i in range(self.n_agents)])  # log all agents' states
        # requests in the next time slot
        self.requests_next = []
        # number of requests in each time slot
        self.n_requests = 0

        # number of steps
        self.n_steps = 0
        self.max_steps = 0

        self.valid_content_indices = []
        self.legal_actions_mask = None

        # return all entities in the world

    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts, no use for now
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update requests state of the world
    def step(self):
        # update requests
        self._deploy_requests(self.requests_next, self.agents)
        # calculate the weighted number of requests
        s = 0
        for request in self.requests_next:
            idx = request.id
            s += self.content_sizes[idx]
        # generate requests of the next time slot
        self.requests_next = generate_requests(num_types=self.n_contents, num_requests=self.n_requests)
        self.n_steps += 1

        return s

    # deploy requests to agents according to their cached contents and locations
    def _deploy_requests(self, requests: list, agents: List[EdgeAgent]):
        for i, agent in enumerate(agents):
            agent.state.request = np.zeros(self.n_contents, dtype=np.int8)
            self.global_state[i].request = agent.state.request

        for request in requests:
            id = request.id
            dis_agent = {}
            for i, agent in enumerate(agents):
                if agent.action.a[id] == 0:
                    continue
                dis = np.linalg.norm(np.array(request.loc) - np.array(agent.location))
                dis_agent[dis] = i

            if dis_agent:
                sorted_dict = dict(sorted(dis_agent.items()))
                agent_id = list(sorted_dict.values())[0]
                agents[agent_id].state.request[id] += 1
                # self.global_state[agent_id].request[id] += 1
