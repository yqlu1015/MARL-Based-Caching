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

MAX_EPISODES = 1500  # 3000
EPISODES_BEFORE_TRAIN = 50
EVAL_EPISODES = 1
EVAL_INTERVAL = 5

MAX_STEPS = 500  # max steps to explore the environment
EVAL_MAX_STEPS = 100
TARGET_UPDATE_INTERVAL = 10  # target net's update interval when using hard update
TARGET_TAU = 0.001  # target net's soft update parameter

N_AGENTS = 5
N_USERS = 3 * N_AGENTS  # reasonable density
N_REQUESTS = 10  # time slot 1h, 10 tasks/s
AGENT_CAPACITY = 2  # content size mean = 1 default=8
AGENT_VIEW = 2
TEMPERATURE = 0.1
REPLACEMENT_FACTOR = 1
ACCURACY_FACTOR = 1

MEMORY_CAPACITY = 100000
BATCH_SIZE = 256
ACTOR_LR = 1e-4
CRITIC_LR = 1e-5
ACTOR_HIDDEN_SIZE = 64
CRITIC_HIDDEN_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
ENTROPY_REG = 0.8

REWARD_DISCOUNTED_GAMMA = 0.95

EPSILON_START = 1.
EPSILON_END = 0.
EPSILON_DECAY = MAX_EPISODES / 4.

DONE_PENALTY = None

RANDOM_SEED = 2023


#  run MARL algorithm with different settings of hyperparameters
def run_params(algo_id: str, algo_handle: Type[Agent], ac=False):
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2
    max_episodes = 1000
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
    batch_sizes = [64, 128, 256]
    rewards_sum = []
    episodes = np.arange(EVAL_INTERVAL, max_episodes + 1, EVAL_INTERVAL)

    start_time = time.time()

    rewards_sum.append([])
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
                           target_update_step=TARGET_UPDATE_INTERVAL)
        eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode

        interact_start_time = time.time()
        while algo.n_episodes < max_episodes:
            algo.interact()
            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                algo.train()

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

        rewards_sum[0].append(eval_rewards_mean_sum)

    time_str = time.strftime('%m_%d_%H_%M')
    data = np.column_stack((episodes, *rewards_sum[0]))
    df = pd.DataFrame(data)
    df.to_csv("./output/%s_%s_param_bs.csv" % (time_str, algo_id), index=False, float_format='%.4f')

    rewards_sum.append([])
    for lr in learning_rates:
        env.reset()
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
                           target_update_step=TARGET_UPDATE_INTERVAL)
        eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode

        interact_start_time = time.time()
        while algo.n_episodes < max_episodes:
            algo.interact()
            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                algo.train()

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

        rewards_sum[1].append(eval_rewards_mean_sum)

    time_str = time.strftime('%m_%d_%H_%M')
    data = np.column_stack((episodes, *rewards_sum[1]))
    df = pd.DataFrame(data)
    df.to_csv("./output/%s_%s_param_lra.csv" % (time_str, algo_id), index=False, float_format='%.4f')

    if ac:
        rewards_sum.append([])
        for lr in learning_rates:
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
                               target_update_step=TARGET_UPDATE_INTERVAL)
            eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode

            interact_start_time = time.time()
            while algo.n_episodes < max_episodes:
                algo.interact()
                if algo.n_episodes >= 20:
                    algo.train()

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

            rewards_sum[2].append(eval_rewards_mean_sum)

        time_str = time.strftime('%m_%d_%H_%M')
        data = np.column_stack((episodes, *rewards_sum[2]))
        df = pd.DataFrame(data)
        df.to_csv("./output/%s_%s_param_lrc.csv" % (time_str, algo_id), index=False, float_format='%.4f')

    end_time = time.time()
    total_time = end_time - start_time

    time_str = time.strftime('%m_%d_%H_%M')
    plot_rewards_sum(episodes, rewards_sum, time_str, algo_id, learning_rates, batch_sizes)
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
                           target_update_step=TARGET_UPDATE_INTERVAL)
        random.seed(RANDOM_SEED)

        interact_start_time = time.time()
        while algo.n_episodes < MAX_EPISODES:
            algo.interact()

            if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                episode_seed = random.randint(0, 2023)
                interact_end_time = time.time()
                rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1, episode_seed)
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

            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
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
                           target_update_step=TARGET_UPDATE_INTERVAL)
        random.seed(RANDOM_SEED)

        interact_start_time = time.time()
        while algo.n_episodes < MAX_EPISODES:
            algo.interact()

            if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                episode_seed = random.randint(0, 2023)
                interact_end_time = time.time()
                rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env_eval, 1, episode_seed)
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

            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
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
    max_episodes = 2000
    episodes_before_train = 100
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2

    average_delays = []  # average delays of the requests of every type of model in some episodes
    cache_hit_ratio = []  # overall cache hit ratio
    episodes = np.arange(EVAL_INTERVAL, max_episodes + 1, EVAL_INTERVAL)
    rewards_sum = []
    cache_stats = []

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
        cache_stat = []
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
                           target_update_step=TARGET_UPDATE_INTERVAL)
        random.seed(RANDOM_SEED)

        train_time = eval_time = 0.
        interact_start_time = start_time = time.time()
        while algo.n_episodes < max_episodes:
            algo.interact()

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
                if algo.n_episodes == MAX_EPISODES:
                    cache_stat = np.mean(stats['cache'], axis=0)

                train_time_epi = interact_end_time - interact_start_time
                train_time += train_time_epi
                eval_time_epi = eval_end_time - interact_end_time
                eval_time += eval_time_epi
                print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                      % (algo.n_episodes, rewards_mu_sum, train_time_epi, eval_time_epi))
                for info in infos:
                    print(info)
                interact_start_time = time.time()

            if algo.n_episodes >= episodes_before_train:
                algo.train()

        total_time = time.time() - start_time
        time_str = time.strftime('%m_%d_%H_%M')
        d = np.array(average_delays[i])
        c = np.array(cache_hit_ratio[i])
        r = np.array(rewards_sum[i])
        data = np.column_stack((episodes, r, d, c))
        cols = ['episode', 'reward sum', 'average delay', 'cache hit ratio']
        df = pd.DataFrame(data, columns=cols)
        df.to_csv("./output/%s_%s_data.csv" % (time_str, algo_id), index=False, float_format='%.4f')
        cache_stats.append(cache_stat)
        save_parameters(algo_id=algo_id, time_str=time_str, method='comp', train_time=train_time, eval_time=eval_time,
                        total_time=total_time, n_agent=n_agents, user_density=3, agent_view=agent_view,
                        max_episodes=max_episodes, episodes_before_train=episodes_before_train)

    # time_str = time.strftime('%m_%d_%H_%M')
    # plot_delay(average_delays, cache_hit_ratio, episodes, time_str, algo_ids)


def run_comp_change(algo_id: str, algo_handle: Type[Agent]):
    max_episodes = 3000
    episodes_before_train = 50
    n_agents = 5
    n_users = 3 * n_agents
    agent_view = 2

    average_delays = []  # average delays of the requests of every type of model in some episodes
    cache_hit_ratio = []  # overall cache hit ratio
    episodes = np.arange(EVAL_INTERVAL, max_episodes + 1, EVAL_INTERVAL)
    rewards_sum = []

    envs = [EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                              agent_view=agent_view,
                              n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                              eta=REPLACEMENT_FACTOR,
                              beta=ACCURACY_FACTOR, seed=RANDOM_SEED, add_ppl=False),
            EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                              agent_view=agent_view,
                              n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                              eta=REPLACEMENT_FACTOR,
                              beta=ACCURACY_FACTOR, seed=RANDOM_SEED)]
    envs_eval = [EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                                   agent_view=agent_view,
                                   n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                   eta=REPLACEMENT_FACTOR,
                                   beta=ACCURACY_FACTOR, seed=RANDOM_SEED, add_ppl=False),
                 EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY,
                                   agent_view=agent_view,
                                   n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                   eta=REPLACEMENT_FACTOR,
                                   beta=ACCURACY_FACTOR, seed=RANDOM_SEED)]
    states_dim = [envs[0].n_models, envs[1].n_models * 2]
    suffix = ['np', 'p']

    for i in range(2):
        state_dim = states_dim[i]
        action_dim = envs[i].n_actions
        actor_output_act = envs[i].action_values_softmax
        critic_output_act = identity

        average_delays.append([])
        cache_hit_ratio.append([])
        rewards_sum.append([])
        algo = algo_handle(env=envs[i], state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=max_episodes / 4., target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL)
        random.seed(RANDOM_SEED)

        train_time = eval_time = 0.
        interact_start_time = start_time = time.time()
        while algo.n_episodes < max_episodes:
            if algo.n_episodes == max_episodes // 2:
                envs[i].world.request_popularity = np.tile(np.arange(envs[i].n_models - 1, -1, -1), (n_users, 1))
                envs_eval[i].world.request_popularity = np.tile(np.arange(envs_eval[i].n_models - 1, -1, -1),
                                                                (n_users, 1))
            algo.interact()

            if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                episode_seed = random.randint(0, 2023)
                interact_end_time = time.time()
                rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(envs_eval[i], 1, episode_seed)
                eval_end_time = time.time()
                # collect infos
                n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                rewards_mu_sum = np.sum(n_rewards_mu)
                rewards_sum[i].append(rewards_mu_sum)
                delay = np.mean(stats['delay'], axis=0)
                average_delays[i].append(delay)
                hit_ratio = np.mean(stats['ratio'], axis=0)
                cache_hit_ratio[i].append(hit_ratio)
                if algo.n_episodes % 500 == 0 and i == 1:
                    time_str = time.strftime('%m_%d_%H_%M')
                    plot_sim_dis(envs_eval[i], time_str, str(algo.n_episodes))

                train_time_epi = interact_end_time - interact_start_time
                train_time += train_time_epi
                eval_time_epi = eval_end_time - interact_end_time
                eval_time += eval_time_epi
                print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                      % (algo.n_episodes, rewards_mu_sum, train_time_epi, eval_time_epi))
                for info in infos:
                    print(info)
                interact_start_time = time.time()

            if algo.n_episodes >= episodes_before_train:
                algo.train()

        total_time = time.time() - start_time
        time_str = time.strftime('%m_%d_%H_%M')
        d = np.array(average_delays[i])
        c = np.array(cache_hit_ratio[i])
        r = np.array(rewards_sum[i])
        data = np.column_stack((episodes, r, d, c))
        cols = ['episode', 'reward sum', 'average delay', 'cache hit ratio']
        df = pd.DataFrame(data, columns=cols)
        df.to_csv("./output/%s_%s_change_%s.csv" % (time_str, algo_id, suffix[i]), index=False, float_format='%.4f')
        save_parameters(algo_id=algo_id, time_str=time_str, method='comp', train_time=train_time, eval_time=eval_time,
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
