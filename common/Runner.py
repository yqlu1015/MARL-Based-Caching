import random
from typing import Type, List
import time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from common.Plot import draw_loc, plot_delay, plot_cache_popularity, plot_rewards_sum, plot_sim_dis

from common.utils import agg_double_list, mean_mean_list, identity
from common.environment.Environment import EdgeMultiAgentEnv, models
from common.Agent import Agent
from algorithms.base import Base

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

EPISODES_BEFORE_TRAIN = 100
MAX_EPISODES = 3000 + EPISODES_BEFORE_TRAIN  # 3000
EVAL_EPISODES = 3
EVAL_INTERVAL = 5

MAX_STEPS = 500  # max steps to explore the environment
EVAL_MAX_STEPS = 100
TARGET_UPDATE_INTERVAL = 10  # target net's update interval when using hard update
TARGET_TAU = 0.01  # target net's soft update parameter, default 1e-3

N_AGENTS = 5
N_USERS = 3 * N_AGENTS  # reasonable density
N_REQUESTS = 10  # time slot 1h, 10 tasks/s
AGENT_CAPACITY = 1.6  # content size mean = 1 default=1.6
AGENT_VIEW = 2
TEMPERATURE = 0.1
REPLACEMENT_FACTOR = 1
ACCURACY_FACTOR = 1

MEMORY_CAPACITY = 500000
BATCH_SIZE = 64  # 256 for q, 64 for ac
ACTOR_LR = 1e-5
CRITIC_LR = 1e-4
ACTOR_HIDDEN_SIZE = 128
CRITIC_HIDDEN_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
ENTROPY_REG = 0.8

REWARD_DISCOUNTED_GAMMA = 0.9

EPSILON_START = 1.
EPSILON_END = 0.
EPSILON_DECAY = MAX_EPISODES / 4.

DONE_PENALTY = None

RANDOM_SEED = 2023


#  run MARL algorithm with different settings of hyperparameters
def run_params(algo_id: str, algo_handle: Type[Agent], lrc=True, lra=False):
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2
    max_episodes = 1000 + EPISODES_BEFORE_TRAIN
    env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY, agent_view=agent_view,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                            eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    env_eval = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                                 agent_view=agent_view,
                                 n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                 eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    state_dim = env.n_models * 2
    action_dim = env.n_actions
    actor_output_act = env.action_values_softmax
    critic_output_act = identity

    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [4, 32, 256]  # [64, 128, 256]
    rewards_sum = []
    episodes = np.arange(0, max_episodes - EPISODES_BEFORE_TRAIN + 1, EVAL_INTERVAL)

    start_time = time.time()

    rewards_sum_bs = []
    for bs in batch_sizes:
        env.reset()
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=bs,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=max_episodes / 4., target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
        eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
        np.random.seed(RANDOM_SEED)

        interact_start_time = time.time()
        while algo.n_episodes < max_episodes:
            algo.interact()

            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                    interact_end_time = time.time()
                    rewards, infos, rewards_c, rewards_s, _, _ = algo.evaluation(env_eval, EVAL_EPISODES)
                    eval_end_time = time.time()
                    n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                    rewards_mu_sum = np.sum(n_rewards_mu)
                    train_time = interact_end_time - interact_start_time
                    eval_time = eval_end_time - interact_end_time
                    print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                          % (algo.n_episodes, rewards_mu_sum, train_time, eval_time))

                    eval_rewards_mean_sum.append(rewards_mu_sum)
                    interact_start_time = time.time()

                algo.train()

        rewards_sum_bs.append(eval_rewards_mean_sum)

    time_str = time.strftime('%m_%d_%H_%M')
    data = np.column_stack((episodes, *rewards_sum_bs))
    df = pd.DataFrame(data)
    df.to_csv("./output/%s_%s_param_bs.csv" % (time_str, algo_id), index=False, float_format='%.4f')
    rewards_sum.append(rewards_sum_bs)

    if lrc:
        rewards_sum_lrc = []
        for lr in learning_rates:
            env.reset()
            algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                               device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                               reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                               actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                               actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                               critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=lr,
                               optimizer_type="adam", entropy_reg=ENTROPY_REG,
                               max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                               episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                               epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                               target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
            eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
            np.random.seed(RANDOM_SEED)

            interact_start_time = time.time()
            while algo.n_episodes < max_episodes:
                algo.interact()
                if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                    if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                        interact_end_time = time.time()
                        rewards, infos, rewards_c, rewards_s, _, _ = algo.evaluation(env_eval, EVAL_EPISODES)
                        eval_end_time = time.time()
                        n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                        rewards_mu_sum = np.sum(n_rewards_mu)
                        train_time = interact_end_time - interact_start_time
                        eval_time = eval_end_time - interact_end_time
                        print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                              % (algo.n_episodes, rewards_mu_sum, train_time, eval_time))

                        eval_rewards_mean_sum.append(rewards_mu_sum)
                        interact_start_time = time.time()

                    algo.train()


            rewards_sum_lrc.append(eval_rewards_mean_sum)

        time_str = time.strftime('%m_%d_%H_%M')
        data = np.column_stack((episodes, *rewards_sum_lrc))
        df = pd.DataFrame(data)
        df.to_csv("./output/%s_%s_param_lra.csv" % (time_str, algo_id), index=False, float_format='%.4f')
        rewards_sum.append(rewards_sum_lrc)

    if lra:
        rewards_sum_lra = []
        for lr in learning_rates:
            algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                               device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                               reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                               actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                               actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                               critic_loss=CRITIC_LOSS, actor_lr=lr, critic_lr=CRITIC_LR,
                               optimizer_type="adam", entropy_reg=ENTROPY_REG,
                               max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                               episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                               epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                               target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
            eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
            np.random.seed(RANDOM_SEED)

            interact_start_time = time.time()
            while algo.n_episodes < max_episodes:
                algo.interact()
                if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                    if algo.episode_done and ((algo.n_episodes + 1) % EVAL_INTERVAL == 0):
                        interact_end_time = time.time()
                        rewards, infos, rewards_c, rewards_s, _, _ = algo.evaluation(env, EVAL_EPISODES)
                        eval_end_time = time.time()
                        n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                        rewards_mu_sum = np.sum(n_rewards_mu)
                        train_time = interact_end_time - interact_start_time
                        eval_time = eval_end_time - interact_end_time
                        print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                              % (algo.n_episodes + 1, rewards_mu_sum, train_time, eval_time))

                        eval_rewards_mean_sum.append(rewards_mu_sum)
                        interact_start_time = time.time()

                    algo.train()

            rewards_sum_lra.append(eval_rewards_mean_sum)

        time_str = time.strftime('%m_%d_%H_%M')
        data = np.column_stack((episodes, *rewards_sum_lra))
        df = pd.DataFrame(data)
        df.to_csv("./output/%s_%s_param_lra.csv" % (time_str, algo_id), index=False, float_format='%.4f')
        rewards_sum.append(rewards_sum_lra)

    end_time = time.time()
    total_time = end_time - start_time

    time_str = time.strftime('%m_%d_%H_%M')
    save_parameters(algo_id=algo_id, time_str=time_str, method='param', train_time=0., eval_time=0.,
                    total_time=total_time, n_agent=n_agents, user_density=3, agent_view=agent_view,
                    max_episodes=max_episodes)


# run MARL algorithm with different zipf parameters for the request distribution
# compare the numbers of cached models
def run_cache(algo_id: str, algo_handle: Type[Agent]):
    zipf_parameters = [0.5, 2.]
    cache_stats_param = []  # average numbers of models caching in the last episode
    spaces = [1.5, 3.5]
    cache_stats_space = []

    start_time = time.time()
    for i, param in enumerate(zipf_parameters):
        env = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY,
                                agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                temperature=TEMPERATURE,
                                eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED, zipf_param=param)
        env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY,
                                     agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                     temperature=TEMPERATURE,
                                     eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED, zipf_param=param)
        print(env.world.model_sizes)
        state_dim = env.n_models * 2
        action_dim = env.n_actions
        actor_output_act = env.action_values_softmax
        critic_output_act = identity
        cache_stats = []
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=MAX_EPISODES)
        np.random.seed(RANDOM_SEED)

        interact_start_time = time.time()
        while algo.n_episodes < MAX_EPISODES:
            algo.interact()

            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                    interact_end_time = time.time()
                    rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1)
                    eval_end_time = time.time()

                    n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                    rewards_mu_sum = np.sum(n_rewards_mu)
                    if algo.n_episodes == MAX_EPISODES:
                        cache_stats = np.mean(stats['cache'], axis=0)

                    train_time = interact_end_time - interact_start_time
                    eval_time = eval_end_time - interact_end_time
                    print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                          % (algo.n_episodes, rewards_mu_sum, train_time, eval_time))
                    for info in infos:
                        print(info)
                    interact_start_time = time.time()

                algo.train()

        cache_stats_param.append(cache_stats)

    for i, space in enumerate(spaces):
        env = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=space,
                                agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                temperature=TEMPERATURE,
                                eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=space,
                                     agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                     temperature=TEMPERATURE,
                                     eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        state_dim = env.n_models * 2
        action_dim = env.n_actions
        actor_output_act = env.action_values_softmax
        critic_output_act = identity
        cache_stats = []
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=MAX_EPISODES)
        np.random.seed(RANDOM_SEED)

        interact_start_time = time.time()
        while algo.n_episodes < MAX_EPISODES:
            algo.interact()

            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                    interact_end_time = time.time()
                    rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1)
                    eval_end_time = time.time()

                    n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                    rewards_mu_sum = np.sum(n_rewards_mu)
                    if algo.n_episodes == MAX_EPISODES:
                        cache_stats = np.mean(stats['cache'], axis=0)

                    train_time = interact_end_time - interact_start_time
                    eval_time = eval_end_time - interact_end_time
                    print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                          % (algo.n_episodes, rewards_mu_sum, train_time, eval_time))
                    for info in infos:
                        print(info)
                    interact_start_time = time.time()

                algo.train()

        cache_stats_space.append(cache_stats)

    total_time = time.time() - start_time
    time_str = time.strftime('%m_%d_%H_%M')
    cols = ['num_cache_param', 'num_cache_space']
    data = np.column_stack((cache_stats_param, cache_stats_space))
    df = pd.DataFrame(data, columns=cols)
    df.to_csv("./output/%s_%s_cache_stats.csv" % (time_str, algo_id), index=False, float_format='%.4f')

    plot_cache_popularity(models, cache_stats_param, time_str, algo_id, params=zipf_parameters)
    plot_cache_popularity(models, cache_stats_space, time_str, algo_id, spaces=spaces)
    save_parameters(algo_id=algo_id, time_str=time_str, method='cache', train_time=0., eval_time=0.,
                    total_time=total_time, n_agent=N_AGENTS, user_density=3, agent_view=AGENT_VIEW,
                    max_episodes=MAX_EPISODES)


# compare convergence of several algorithms
def run_comp(algo_ids: List[str], algo_handles: List[Type[Agent]]):
    max_episodes = 3100
    episodes_before_train = 100
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2  # 4 if n_agents=10
    agent_capacity = AGENT_CAPACITY  # 1.5 if n_agents=10
    episodes = np.arange(0, max_episodes - episodes_before_train + 1, EVAL_INTERVAL)

    for i, (algo_id, algo_handle) in enumerate(zip(algo_ids, algo_handles)):
        average_delays = []  # average delays of the requests of every type of model in some episodes
        cache_hit_ratio = []  # overall cache hit ratio
        agent_rewards = []
        rewards_sum = []

        env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity,
                                agent_view=agent_view,
                                n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                eta=REPLACEMENT_FACTOR,
                                beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        env_eval = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity,
                                     agent_view=agent_view,
                                     n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                     eta=REPLACEMENT_FACTOR,
                                     beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        print(env.world.model_sizes)

        state_dim = env.n_models * 2
        action_dim = env.n_actions
        actor_output_act = env.action_values_softmax
        critic_output_act = identity

        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=max_episodes / 4., target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
        np.random.seed(RANDOM_SEED)

        train_time = eval_time = 0.
        interact_start_time = start_time = time.time()
        while algo.n_episodes < max_episodes:
            algo.interact()

            if algo.n_episodes >= episodes_before_train:
                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                    interact_end_time = time.time()
                    rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1)
                    eval_end_time = time.time()
                    # collect infos
                    n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                    agent_rewards.append(n_rewards_mu)
                    rewards_mu_sum = np.sum(n_rewards_mu)
                    rewards_sum.append(rewards_mu_sum)
                    delay = np.mean(stats['delay'], axis=0)
                    average_delays.append(delay)
                    hit_ratio = np.mean(stats['ratio'], axis=0)
                    cache_hit_ratio.append(hit_ratio)

                    train_time_epi = interact_end_time - interact_start_time
                    train_time += train_time_epi
                    eval_time_epi = eval_end_time - interact_end_time
                    eval_time += eval_time_epi
                    print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                          % (algo.n_episodes, rewards_mu_sum, train_time_epi, eval_time_epi))
                    for info in infos:
                        print(info)
                    interact_start_time = time.time()

                algo.train()

        total_time = time.time() - start_time
        time_str = time.strftime('%m_%d_%H_%M')
        d = np.array(average_delays)
        c = np.array(cache_hit_ratio)
        r = np.array(rewards_sum)
        a_r = np.array(agent_rewards)
        erdc = np.column_stack((episodes, r, d, c))
        data = np.c_[erdc, a_r]
        cols = ['episode', 'reward sum', 'average delay', 'cache hit ratio']
        cols.extend([f"reward_{i}" for i in range(n_agents)])
        df = pd.DataFrame(data, columns=cols)
        df.to_csv("./output/%s_%s_data.csv" % (time_str, algo_id), index=False, float_format='%.4f')
        save_parameters(algo_id=algo_id, time_str=time_str, method='comp', train_time=train_time, eval_time=eval_time,
                        total_time=total_time, n_agent=n_agents, user_density=3, agent_view=agent_view,
                        max_episodes=max_episodes, episodes_before_train=episodes_before_train)

    # time_str = time.strftime('%m_%d_%H_%M')
    # plot_delay(average_delays, cache_hit_ratio, episodes, time_str, algo_ids)


def run_comp_change(algo_ids: List[str], algo_handles: List[Type[Agent]]):
    max_episodes = 3100
    episodes_before_train = 100
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2

    average_delays = []  # average delays of the requests of every type of model in some episodes
    cache_hit_ratio = []  # overall cache hit ratio
    episodes = np.arange(0, max_episodes - episodes_before_train + 1, EVAL_INTERVAL)
    rewards_sum = []
    suffix = ['np', 'p']

    for i, (algo_id, algo_handle) in enumerate(zip(algo_ids, algo_handles)):
        env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                                agent_view=agent_view,
                                n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                eta=REPLACEMENT_FACTOR,
                                beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        env_eval = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                                     agent_view=agent_view,
                                     n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                     eta=REPLACEMENT_FACTOR,
                                     beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        state_dim = env.n_models * 2
        action_dim = env.n_actions
        actor_output_act = env.action_values_softmax
        critic_output_act = identity

        average_delays.append([])
        cache_hit_ratio.append([])
        rewards_sum.append([])
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=max_episodes / 4., target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
        np.random.seed(RANDOM_SEED)

        train_time = eval_time = 0.
        interact_start_time = start_time = time.time()
        while algo.n_episodes < max_episodes:
            if algo.n_episodes == 1000 or algo.n_episodes == 2000:
                env.world.request_popularity = np.tile(np.random.permutation(env.n_models), (n_users, 1))
                env_eval.world.request_popularity = env.world.request_popularity
                print(f"changed popularity: {env.world.request_popularity[0]}")
            algo.interact()

            if algo.n_episodes >= episodes_before_train:
                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                    episode_seed = random.randint(0, 2023)
                    interact_end_time = time.time()
                    rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1, episode_seed)
                    eval_end_time = time.time()
                    # collect infos
                    n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                    rewards_mu_sum = np.sum(n_rewards_mu)
                    rewards_sum[i].append(rewards_mu_sum)
                    delay = np.mean(stats['delay'], axis=0)
                    average_delays[i].append(delay)
                    hit_ratio = np.mean(stats['ratio'], axis=0)
                    cache_hit_ratio[i].append(hit_ratio)
                    # if algo.n_episodes % 500 == 0:
                    #     time_str = time.strftime('%m_%d_%H_%M')
                    #     plot_sim_dis(env_eval, time_str, str(algo.n_episodes))

                    train_time_epi = interact_end_time - interact_start_time
                    train_time += train_time_epi
                    eval_time_epi = eval_end_time - interact_end_time
                    eval_time += eval_time_epi
                    print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                          % (algo.n_episodes, rewards_mu_sum, train_time_epi, eval_time_epi))
                    for info in infos:
                        print(info)
                    interact_start_time = time.time()

                algo.train()

        total_time = time.time() - start_time
        time_str = time.strftime('%m_%d_%H_%M')
        d = np.array(average_delays[i])
        c = np.array(cache_hit_ratio[i])
        r = np.array(rewards_sum[i])
        data = np.column_stack((episodes, r, d, c))
        cols = ['episode', 'reward sum', 'average delay', 'cache hit ratio']
        df = pd.DataFrame(data, columns=cols)
        df.to_csv("./output/%s_%s_change.csv" % (time_str, algo_id), index=False, float_format='%.4f')
        save_parameters(algo_id=algo_id, time_str=time_str, method='change', train_time=train_time, eval_time=eval_time,
                        total_time=total_time, n_agent=n_agents, user_density=3, agent_view=agent_view,
                        max_episodes=max_episodes, episodes_before_train=episodes_before_train)


def save_parameters(algo_id: str, time_str: str, method: str, train_time: float, eval_time: float, total_time: float,
                    n_agent: int, user_density: int, agent_view: int, max_episodes: int,
                    episodes_before_train=EPISODES_BEFORE_TRAIN):
    parameters = f"algo_id: {algo_id}, random seed: {RANDOM_SEED},\n" \
                 f"n_agents: {n_agent},\nuser_density: {user_density},\n" \
                 f"n_requests: {N_REQUESTS},\n" \
                 f"agent_capacity: {AGENT_CAPACITY},\nagent_view: {agent_view},\n" \
                 f"replacement factor: {REPLACEMENT_FACTOR}, accuracy factor: {ACCURACY_FACTOR}\n" \
                 f"actor_hidden_size: {ACTOR_HIDDEN_SIZE}, critic_hidden_size: {CRITIC_HIDDEN_SIZE},\n" \
                 f"batch size: {BATCH_SIZE}, critic loss: {CRITIC_LOSS},\n" \
                 f"episodes number: {max_episodes}, episodes before training: {episodes_before_train},\n" \
                 f"evaluation interval: {EVAL_INTERVAL}, evaluation episodes number: {EVAL_EPISODES},\n" \
                 f"max steps: {MAX_STEPS}, evaluation_max_steps: {EVAL_MAX_STEPS},\n" \
                 f"temperature: {TEMPERATURE},\n" \
                 f"training time: {train_time: .4f} s,\n" \
                 f"evaluation time: {eval_time: .4f} s, \n" \
                 f"total time: {total_time: .4f} s"
    f = open(f"./output/{time_str}_{algo_id}_{method}.txt", "w")
    f.write(parameters)
    f.close()
