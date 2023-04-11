import numpy as np
from typing import List

from common.environment.utils.distribution import generate_requests


class EdgeState(object):
    def __init__(self):
        # self.request = []  # numbers of requests of each dnn
        self.popularity = []
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
        # -1: observe the whole state, 0: itself, n_agents: cloeset n_agents neighbors
        self.view_sight = view_sight
        self.neighbor_mask = None  # the mask for who are neighbors
        # action
        self.action = EdgeAction()
        # state
        self.state = EdgeState()
        # script behavior to execute
        self.action_callback = None
        # reward for cache hit
        self.reward_c = 0
        # switch cost
        self.reward_s = 0


# DNN model
class Model(object):
    def __init__(self, name: str, type: str, accuracy: float, model_size: float, time_c: float, time_e: float,
                 input_size: float):
        self.name = name
        self.type = type  # the same type of dnn models share the same type_id, e.g. resnet18, resnet34
        self.accuracy = accuracy  # top-1 accuracy
        self.size = model_size  # MB
        self.inf_time_c = time_c  # s
        self.inf_time_e = time_e  # s
        self.input_size = input_size  # Mb


# IoT device as user in this scenario
class User(object):
    def __init__(self, user_id, loc=np.array([0, 0]), models_num=10):
        self.id = user_id
        self.loc = loc
        self.target = np.zeros(models_num, dtype='int')
        self.accuracy = np.zeros(models_num, dtype=np.float32)  # %
        self.trans_delay = np.zeros(models_num, dtype=np.float32)  # s
        self.inf_delay = np.zeros(models_num, dtype=np.float32)  # s


class EdgeWorld(object):
    def __init__(self, n_agents=10, n_users=30):
        # shape of the world, i.e. area covered by all edges
        self.shape_size = [0, 0]

        # list of agents (can change at execution-time!)
        self.agents: List[EdgeAgent] = []
        self.n_agents = n_agents
        self.agent_view_sight = 1
        self.agent_storage = 0

        # list of dnn models
        self.models: List[List[Model]] = []
        self.n_model_types = 0
        self.n_models = 0
        self.model_sizes = []
        self.model_input_sizes = []

        # list of iot devices
        self.users: List[User] = []
        self.n_users = n_users

        # transmission rate from IoTDs to edge servers
        self.trans_rates_ie = np.zeros((self.n_users, self.n_agents))
        # transmission rate from edges to the cloud
        self.trans_rates_ec = np.zeros(self.n_agents)

        # state information
        self.global_state: np.ndarray[EdgeState] = np.array([EdgeState() for _ in range(n_agents)])  # log all agents' states

        self.n_requests = 0  # number of requests of each user in each time slot
        self.request_popularity = None  # indices of users' requested models sorted by popularity in descending order
        # requests in the next time slot
        self.requests_next: np.array = None

        # number of steps
        self.n_steps = 0
        self.max_steps = 0

        # 1-to-1 mapping between arrays of valid cache and integers as discrete action in RL
        self.number2cache = []
        self.n_caches = 0
        self.cache_stat: np.array = None  # number of caches of each model in one episode
        self.average_delay = 0.  # average delay of each model type, each user
        self.cache_hit_ratio = 0.  # defined as (# requests served by edges/ # requests)

        self.zipf_param = 0.8

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
    def step(self, beta=1):
        # update models popularity and cached models at each edge
        self._update_popularity(beta)
        self._update_stat()
        self._update_cache()
        # generate requests of the next time slot
        self.requests_next = generate_requests(num_users=self.n_users, num_types=self.n_model_types,
                                               num_low=int(self.n_requests*0.8), num_high=int(self.n_requests*1.2),
                                               orders=self.request_popularity, zipf_param=self.zipf_param)
        self.n_steps += 1

    # update the popularity by requests and edges' action
    def _update_popularity(self, beta=1):
        # clear the popularity and reward in the last time slot
        for m, s in enumerate(self.global_state):
            s.popularity = np.zeros(self.n_model_types)
            self.agents[m].reward_s = 0
            self.agents[m].reward_c = 0
        self.cache_hit_ratio = 0.

        # users determine offloading targets
        for type_idx in range(self.n_model_types):  # j
            for n, user in enumerate(self.users):  # n_agents
                edge_reward = {}
                edge_acc = {}
                edge_trans = {}
                edge_inf = {}
                for m, edge in enumerate(self.agents):  # b_{nj}
                    model_idx = edge.action.a[type_idx]
                    if model_idx > 0:
                        model = self.models[type_idx][model_idx-1]  # b_{nj}j
                        trans_delay = model.input_size / self.trans_rates_ie[n][m]
                        inf_delay = model.inf_time_e
                    else:
                        model = self.models[type_idx][0]
                        trans_delay = model.input_size * (1/self.trans_rates_ie[n][m]+1/self.trans_rates_ec[m])
                        inf_delay = model.inf_time_c
                    acc = model.accuracy
                    edge_reward[m] = beta * acc - (trans_delay + inf_delay)
                    edge_acc[m] = acc
                    edge_trans[m] = trans_delay
                    edge_inf[m] = inf_delay
                edge_idx = max(edge_reward, key=edge_reward.get)
                user.target[type_idx] = edge_idx
                user.accuracy[type_idx] = edge_acc[edge_idx]
                user.trans_delay[type_idx] = edge_trans[edge_idx]
                user.inf_delay[type_idx] = edge_inf[edge_idx]

                # calculate revenue of cache hit
                agent_idx = user.target[type_idx]
                self.global_state[agent_idx].popularity[type_idx] += self.requests_next[n][type_idx]

        # calculate rewards and update popularity
        for m, edge in enumerate(self.agents):
            edge.reward_c = sum(self.global_state[m].popularity * (edge.action.a != 0) * self.model_input_sizes) / 1000
            self.cache_hit_ratio += sum(self.global_state[m].popularity * (edge.action.a != 0))
            norm = np.linalg.norm(self.global_state[m].popularity, 1)
            if norm != 0:
                self.global_state[m].popularity = self.global_state[m].popularity / norm

            for type_idx in range(self.n_model_types):
                model_idx = edge.action.a[type_idx]
                if model_idx != edge.state.cache[type_idx] and model_idx != 0:
                    edge.reward_s += self.model_sizes[type_idx][model_idx-1]

        self.cache_hit_ratio /= sum(sum(self.requests_next))

    # update cached dnns for a particular agent
    def _update_cache(self):
        for m, s in enumerate(self.global_state):
            s.cache = self.agents[m].action.a

    # update the statistics of model caches and average delay
    def _update_stat(self):
        self.cache_stat = np.zeros(self.n_models, dtype='int')
        for edge in self.agents:
            pre_cnt = 0
            for i, cache in enumerate(edge.action.a):
                if cache != 0:
                    self.cache_stat[pre_cnt+cache-1] += 1
                pre_cnt += len(self.models[i])

        self.average_delay = 0.
        for i, user in enumerate(self.users):
            for type_idx in range(self.n_model_types):
                self.average_delay += (user.inf_delay[type_idx] + user.trans_delay[type_idx]) * \
                                      self.requests_next[i][type_idx]

        self.average_delay /= sum(sum(self.requests_next))
