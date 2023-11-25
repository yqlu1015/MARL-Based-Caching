import numpy as np
from typing import List

from common.environment.utils.distribution import generate_requests


class EdgeState(object):
    def __init__(self):
        # self.request = []  # numbers of requests of each dnn
        self.popularity: np.ndarray = []
        self.cache: np.ndarray = []  # one-hot vector of cached dnns


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
        self.switch = 0
        # average offloading qoe of tasks processed by it
        self.avg_qoe = 0.
        self.avg_delay_old = 0.


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
        self.requests_next = None
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
        self.observations: np.ndarray[EdgeState] = np.array(
            [EdgeState() for _ in range(n_agents)])  # all agents' observation
        self.global_popularity: np.ndarray = np.zeros(10)

        self.n_requests = 0  # number of requests of each user in each time slot
        self.request_popularity = None  # indices of users' requested models sorted by popularity in descending order
        # requests in the next time slot

        # number of steps
        self.n_steps = 0
        self.max_steps = 0

        # 1-to-1 mapping between arrays of valid cache and integers as discrete action in RL
        self.number2cache = []
        self.cache2number = {}
        self.n_caches = 0
        self.cache_stat: np.array = None  # number of caches of each model in one episode
        self.average_delay = 0.  # average delay of each model type, each user
        self.average_delay_old = 0.
        self.popularity_old = None  # 2d numpy array[n_agents x n_model_types], popularity if old cache
        self.cache_hit_ratio = 0.  # defined as (# requests served by edges/ # requests)
        self.avg_reward_c = 0.
        self.switch_sum = 0.
        self.avg_qoe = 0.  # average quality of experience of users

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
                                               num_low=int(self.n_requests * 0.8), num_high=int(self.n_requests * 1.2),
                                               orders=self.request_popularity, zipf_param=self.zipf_param)
        # self._update_global_popularity()
        self.n_steps += 1

    # update the popularity by requests and edges' action
    def _update_popularity(self, beta=1):
        # clear the popularity and reward in the last time slot
        for m, s in enumerate(self.observations):
            s.popularity = np.zeros(self.n_model_types)
            self.agents[m].switch = 0
            # self.agents[m].reward_c = 0
            self.agents[m].avg_qoe = 0.
            # self.agents[m].avg_delay_old = 0.
        self.cache_hit_ratio = 0.
        self.average_delay = 0.
        self.avg_qoe = 0.
        self.switch_sum = 0
        # self.average_delay_old = 0.
        # self.popularity_old = np.zeros((self.n_agents, self.n_model_types))

        # users determine offloading targets
        for type_idx in range(self.n_model_types):  # j
            for n, user in enumerate(self.users):  # n_agents
                edge_reward = {}
                edge_acc = {}
                edge_trans = {}
                edge_inf = {}
                # edge_reward_old = {}  # payoff of USER if the old cache
                # edge_trans_old = {}
                # edge_inf_old = {}
                for m, edge in enumerate(self.agents):  # b_{nj}
                    self._calc_user_payoff(action=edge.action.a, user_idx=n, agent_idx=m, type_idx=type_idx, beta=beta,
                                           edge_reward=edge_reward,
                                           edge_acc=edge_acc, edge_trans=edge_trans, edge_inf=edge_inf)
                    # self._calc_user_payoff(action=self.observations[m].cache, user_idx=n, agent_idx=m,
                    #                        type_idx=type_idx, beta=beta, edge_reward=edge_reward_old,
                    #                        edge_trans=edge_trans_old, edge_inf=edge_inf_old)
                edge_idx = max(edge_reward, key=edge_reward.get)
                user.target[type_idx] = edge_idx
                user.accuracy[type_idx] = edge_acc[edge_idx]
                user.trans_delay[type_idx] = edge_trans[edge_idx]
                user.inf_delay[type_idx] = edge_inf[edge_idx]
                # edge_idx_old = max(edge_reward_old, key=edge_reward_old.get)

                # add total offloading delay of user N for task TYPE_IDX to the global and AGENT_IDX's average delay
                # add requests number of user N for task TYPE_IDX to AGENT_IDX's popularity
                off_delay = edge_inf[edge_idx] + edge_trans[edge_idx]
                qoe = beta * edge_acc[edge_idx] - off_delay
                self.agents[edge_idx].avg_qoe += qoe * self.requests_next[n][type_idx]
                self.average_delay += off_delay * self.requests_next[n][type_idx]
                self.avg_qoe += qoe * self.requests_next[n][type_idx]
                self.observations[edge_idx].popularity[type_idx] += self.requests_next[n][type_idx]
                # off_delay_old = edge_inf_old[edge_idx_old] + edge_trans_old[edge_idx_old]
                # self.agents[edge_idx].avg_delay_old += off_delay_old * self.requests_next[n][type_idx]
                # self.average_delay_old += off_delay_old * self.requests_next[n][type_idx]

                # self.popularity_old[edge_idx_old, type_idx] += self.requests_next[n][type_idx]

        self.average_delay /= np.sum(self.requests_next)
        self.avg_qoe /= np.sum(self.requests_next)
        # self.average_delay_old /= sum(sum(self.requests_next))

        # calculate rewards and update popularity
        for m, edge in enumerate(self.agents):
            # cached_model_sizes = np.zeros(self.n_model_types)
            for type_idx in range(self.n_model_types):
                model_idx = edge.action.a[type_idx]
                if model_idx != 0 and model_idx != self.observations[m].cache[type_idx]:
                    # cached_model_sizes[type_idx] = self.model_sizes[type_idx][model_idx - 1]
                    edge.switch += self.model_sizes[type_idx][model_idx - 1]
            self.switch_sum += edge.switch

            # edge.reward_c = sum(self.observations[m].popularity * cached_model_sizes)
            ppl_sum = np.sum(self.observations[m].popularity)
            edge.avg_qoe = edge.avg_qoe / ppl_sum if ppl_sum != 0 else 0
            # ppl_sum_old = sum(self.popularity_old[m])
            # edge.avg_delay_old = edge.avg_delay_old / ppl_sum_old if ppl_sum_old != 0 else 1
            self.cache_hit_ratio += np.sum(self.observations[m].popularity * (edge.action.a != 0))
            norm = np.linalg.norm(self.observations[m].popularity, 1)
            if norm != 0:
                self.observations[m].popularity = self.observations[m].popularity / norm

        self.cache_hit_ratio /= np.sum(self.requests_next)
        self.switch_sum /= self.n_agents

    # update cached dnns for a particular agent
    def _update_cache(self):
        for m, s in enumerate(self.observations):
            s.cache = self.agents[m].action.a

    # update the statistics of model caches and average delay
    def _update_stat(self):
        self.cache_stat = np.zeros(self.n_models, dtype='int')
        # self.avg_reward_c = 0
        for edge in self.agents:
            pre_cnt = 0
            # self.avg_reward_c += edge.reward_c
            for i, cache in enumerate(edge.action.a):
                if cache != 0:
                    self.cache_stat[pre_cnt + cache - 1] += 1
                pre_cnt += len(self.models[i])
        # self.avg_reward_c /= self.n_agents

    # def _update_global_popularity(self):
    #     self.global_popularity = np.sum(self.requests_next, axis=0) / np.sum(self.requests_next)

    # calculate the payoff for a task of a user given an action
    def _calc_user_payoff(self, action: list, user_idx: int, type_idx: int, agent_idx: int, beta=1,
                          edge_reward: dict = {}, edge_acc: dict = None, edge_trans: dict = {}, edge_inf: dict = {}):
        model_idx = action[type_idx]
        if model_idx > 0:
            model = self.models[type_idx][model_idx - 1]  # b_{nj}j
            trans_delay = model.input_size / self.trans_rates_ie[user_idx][agent_idx] * 1000  # ms
            inf_delay = model.inf_time_e  # ms
        else:
            model = self.models[type_idx][0]
            trans_delay = model.input_size * (
                    1 / self.trans_rates_ie[user_idx][agent_idx] + 1 / self.trans_rates_ec[agent_idx]) * 1000  # ms
            inf_delay = model.inf_time_c  # ms
        acc = model.accuracy
        edge_reward[agent_idx] = beta * acc - (trans_delay + inf_delay)
        edge_trans[agent_idx] = trans_delay
        edge_inf[agent_idx] = inf_delay
        if edge_acc is not None:
            edge_acc[agent_idx] = acc
