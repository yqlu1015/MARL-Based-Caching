import random
from typing import Type, List
import time
import numpy as np
import torch as th
import pandas as pd
from common.Plot import draw_loc, plot_delay, plot_cache_popularity, plot_rewards, plot_sim_dis

from common.utils import agg_double_list, mean_mean_list, identity
from common.environment.Environment import EdgeMultiAgentEnv, models
from common.Agent import Agent
from algorithms.base import Base

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

EPISODES_BEFORE_TRAIN = 10
MAX_EPISODES = 4000 + EPISODES_BEFORE_TRAIN  # 3000
EVAL_EPISODES = 1
EVAL_INTERVAL = 10  # default=5

MAX_STEPS = 300  # max steps to explore the environment
EVAL_MAX_STEPS = 100
TARGET_UPDATE_INTERVAL = 10  # target net's update interval when using hard update
TARGET_TAU = 0.01  # target net's soft update parameter, default 1e-3

N_AGENTS = 5
N_USERS = 3 * N_AGENTS  # reasonable density
N_REQUESTS = 10  # time slot 1h, 10 tasks/s
AGENT_CAPACITY = 1.6  # content size mean = 1 default=1.6
AGENT_VIEW = 2
TEMPERATURE = 0.1
QOE_FACTOR = 1
ACCURACY_FACTOR = 1

MEMORY_CAPACITY = 3 * 10 ** 5
BATCH_SIZE = 256  # 256 for q, 64 for ac
ACTOR_LR = 5e-5
CRITIC_LR = 1e-4
ACTOR_HIDDEN_SIZE = 128
CRITIC_HIDDEN_SIZE = 64  # 128 for q
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
ENTROPY_REG = 0.08

REWARD_DISCOUNTED_GAMMA = 0.95

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
    agent_capacity = 1.6
    env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity, agent_view=agent_view,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                            eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    env_eval = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity,
                                 agent_view=agent_view,
                                 n_requests=N_REQUESTS, max_steps=EVAL_MAX_STEPS, temperature=TEMPERATURE,
                                 eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    print(env.n_actions)
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
            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                algo.train()
            algo.interact()

            if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0) and algo.n_episodes >= EPISODES_BEFORE_TRAIN:
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
                if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                    algo.train()

                algo.interact()

                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0) and algo.n_episodes >= EPISODES_BEFORE_TRAIN:
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

            rewards_sum_lrc.append(eval_rewards_mean_sum)

        time_str = time.strftime('%m_%d_%H_%M')
        data = np.column_stack((episodes, *rewards_sum_lrc))
        df = pd.DataFrame(data)
        df.to_csv("./output/%s_%s_param_lrc.csv" % (time_str, algo_id), index=False, float_format='%.4f')
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
                if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                    algo.train()

                algo.interact()

                if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0) and algo.n_episodes >= EPISODES_BEFORE_TRAIN:
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
                                eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED, zipf_param=param)
        env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY,
                                     agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                     temperature=TEMPERATURE,
                                     eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED, zipf_param=param)
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
                                eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
        env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=space,
                                     agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS,
                                     temperature=TEMPERATURE,
                                     eta=QOE_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
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
                if algo.episode_done and (
                        algo.n_episodes % EVAL_INTERVAL == 0) and algo.n_episodes > EPISODES_BEFORE_TRAIN:
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
def run_comp(algo_ids: List[str], algo_handles: List[Type[Agent]], add_ppl=True, change=False, paths=None):
    episodes_before_train = 100
    max_episodes = 3000 + episodes_before_train
    epsilon_decay = (max_episodes - episodes_before_train) / 2.
    n_agents = 10
    n_users = 5 * n_agents
    agent_view = 2  # 4 if n_agents=10
    agent_capacity = 1.6  # 1.5 if n_agents=10
    bs = 64  # 256 for q, 64 for ac
    memory_size = 3 * 10 ** 5
    lra = 1e-5
    lrc = 1e-4
    mid_dim_a = 128
    mid_dim_c = 64  # 128 for q
    zipf_param = 0.8
    # episodes = np.arange(0, max_episodes - episodes_before_train, EVAL_INTERVAL)

    for i, (algo_id, algo_handle) in enumerate(zip(algo_ids, algo_handles)):
        episodes = []
        average_delays = []  # average delays of the requests of every type of model in some episodes
        cache_hit_ratio = []  # overall cache hit ratio
        avg_switch = []  # average switch cost of all agents
        avg_qoe = []
        agent_switches = []
        agent_qoes = []
        rewards_sum = []
        eval_records = []

        env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity,
                                agent_view=agent_view,
                                n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                                eta=QOE_FACTOR,
                                beta=ACCURACY_FACTOR, seed=RANDOM_SEED, add_ppl=add_ppl, zipf_param=zipf_param)
        env_eval = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=agent_capacity,
                                     agent_view=agent_view,
                                     n_requests=N_REQUESTS, max_steps=EVAL_MAX_STEPS, temperature=TEMPERATURE,
                                     eta=QOE_FACTOR,
                                     beta=ACCURACY_FACTOR, seed=RANDOM_SEED, add_ppl=add_ppl, zipf_param=zipf_param)
        print(env.world.model_sizes)
        print(env.n_actions)

        state_dim = env.n_models if not add_ppl else env.n_models * 2
        action_dim = env.n_actions
        actor_output_act = env.action_values_softmax
        critic_output_act = identity

        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=memory_size, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=mid_dim_a, critic_hidden_size=mid_dim_c,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=lra, critic_lr=lrc,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=bs,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=epsilon_decay, target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL, max_episodes=max_episodes)
        # np.random.seed(RANDOM_SEED)
        if paths is not None:
            algo.load_models(paths[i])
        # algo.save_models(algo_id)

        train_time = eval_time = 0.
        interact_start_time = start_time = time.time()
        while algo.n_episodes < max_episodes:
            if algo.n_episodes >= episodes_before_train:
                algo.train()

            if change and algo.n_episodes == 1000 + episodes_before_train:
                env.world.request_popularity = np.tile(np.random.permutation(env.n_models), (n_users, 1))
                env_eval.world.request_popularity = env.world.request_popularity
                print(f"changed popularity: {env.world.request_popularity[0]}")
            algo.interact()

            if algo.episode_done and (algo.n_episodes % EVAL_INTERVAL == 0):
                interact_end_time = time.time()
                rewards, infos, rewards_q, rewards_s, users_info, stats = algo.evaluation(env_eval, EVAL_EPISODES)
                eval_end_time = time.time()
                # collect infos
                n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                n_qoe_mu, n_qoe_std = mean_mean_list(rewards_q)
                n_switch_mu, n_switch_std = mean_mean_list(rewards_s)
                # agent_rewards.append(n_rewards_mu)
                rewards_mu_sum = n_rewards_mu[0]  # np.sum(n_rewards_mu)
                agent_qoes.append([n_qoe_mu, n_qoe_std])
                agent_switches.append([n_switch_mu, n_switch_std])
                rewards_sum.append(rewards_mu_sum)
                delay = np.mean(stats['delay'], axis=0)
                average_delays.append(delay)
                hit_ratio = np.mean(stats['ratio'], axis=0)
                cache_hit_ratio.append(hit_ratio)
                switch = np.mean(stats['switch'], axis=0)
                avg_switch.append(switch)
                qoe = np.mean(stats['qoe'], axis=0)
                avg_qoe.append(qoe)
                eval_records.append(stats['eval_records'])
                episodes.append(algo.n_episodes)

                train_time_epi = interact_end_time - interact_start_time
                train_time += train_time_epi
                eval_time_epi = eval_end_time - interact_end_time
                eval_time += eval_time_epi
                print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                      % (algo.n_episodes, rewards_mu_sum, train_time_epi, eval_time_epi))
                for info in infos:
                    print(info)
                interact_start_time = time.time()

        total_time = time.time() - start_time
        time_str = time.strftime('%m_%d_%H_%M')
        e = np.array(episodes)
        s = np.array(avg_switch)
        d = np.array(average_delays)
        c = np.array(cache_hit_ratio) * 100  # convert from decimal to %
        r = np.array(rewards_sum)
        q = np.array(avg_qoe)
        a_s = np.array(agent_switches)
        a_s_mu = a_s[:, 0]
        a_s_std = a_s[:, 1]
        a_q = np.array(agent_qoes)
        a_q_mu = a_q[:, 0]
        a_q_std = a_q[:, 1]
        erdc = np.column_stack((e, r, d, c, s, q))
        data = np.c_[erdc, a_q_mu, a_q_std, a_s_mu, a_s_std]
        cols = ['episode', 'reward sum', 'average delay', 'cache hit ratio', 'average switch cost', 'average qoe']
        cols.extend([f"{i}_qoe_mu" for i in range(n_agents)])
        cols.extend([f"{i}_qoe_std" for i in range(n_agents)])
        cols.extend([f"{i}_switch_mu" for i in range(n_agents)])
        cols.extend([f"{i}_switch_std" for i in range(n_agents)])
        df = pd.DataFrame(data, columns=cols)
        df.to_csv("./output/%s_%s_data.csv" % (time_str, algo_id), index=False, float_format='%.4f')

        if algo_id != 'random':
            records = np.array(eval_records).reshape(-1, 6)
            extended_episodes = np.repeat(e, EVAL_MAX_STEPS)
            data = np.c_[extended_episodes, records]
            cols = ['episode', 'value 0', 'value 1', 'value 2', 'index 0', 'index 1', 'index 2']
            df = pd.DataFrame(data, columns=cols)
            df.to_csv("./output/%s_%s_actor_values.csv" % (time_str, algo_id), index=False, float_format='%.4f')

        method = 'comp' if not change else 'change'
        save_parameters(algo_id=algo_id, time_str=time_str, method=method, train_time=train_time, eval_time=eval_time,
                        total_time=total_time, n_agent=n_agents, user_density=3, agent_view=agent_view,
                        max_episodes=max_episodes, episodes_before_train=episodes_before_train,
                        epsilon_decay=epsilon_decay, n_model_types=env.n_models, actor_lr=lra, critic_lr=lrc)

        # algo.save_models(algo_id)
        # algo.save_memory(algo_id)

    # time_str = time.strftime('%m_%d_%H_%M')
    # plot_delay(average_delays, cache_hit_ratio, episodes, time_str, algo_ids)


def save_parameters(algo_id: str, time_str: str, method: str, train_time: float, eval_time: float, total_time: float,
                    n_agent: int, user_density: int, agent_view: int, max_episodes: int,
                    episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_decay=EPSILON_DECAY, n_model_types=10,
                    actor_lr=1e-5, critic_lr=1e-4):
    parameters = f"algo_id: {algo_id}, random seed: {RANDOM_SEED},\n" \
                 f"n_agents: {n_agent},\nuser_density: {user_density},\n" \
                 f"n_requests: {N_REQUESTS}, n_model_types: {n_model_types},\n" \
                 f"agent_capacity: {AGENT_CAPACITY},\nagent_view: {agent_view},\n" \
                 f"QoE factor: {QOE_FACTOR}, accuracy factor: {ACCURACY_FACTOR}\n" \
                 f"actor_hidden_size: {ACTOR_HIDDEN_SIZE}, critic_hidden_size: {CRITIC_HIDDEN_SIZE},\n" \
                 f"batch size: {BATCH_SIZE},  memory size: {MEMORY_CAPACITY}, critic loss: {CRITIC_LOSS},\n" \
                 f"episodes number: {max_episodes}, episodes before training: {episodes_before_train}, " \
                 f"epsilon decay: {epsilon_decay}, \n" \
                 f"evaluation interval: {EVAL_INTERVAL}, evaluation episodes number: {EVAL_EPISODES},\n" \
                 f"max steps: {MAX_STEPS}, evaluation_max_steps: {EVAL_MAX_STEPS},\n" \
                 f"temperature: {TEMPERATURE},\n" \
                 f"training time: {train_time: .4f} s,\n" \
                 f"evaluation time: {eval_time: .4f} s, \n" \
                 f"total time: {total_time: .4f} s"
    f = open(f"./output/{time_str}_{algo_id}_{method}.txt", "w")
    f.write(parameters)
    f.close()
