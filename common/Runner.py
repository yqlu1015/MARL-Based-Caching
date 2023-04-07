import random
from typing import Type
import time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from common.utils import agg_double_list, mean_mean_list, identity
from common.environment.Environment import EdgeMultiAgentEnv, action_values_softmax
from common.Agent import Agent
from algorithms.base import Base

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

MAX_EPISODES = 3000  # 3000
EPISODES_BEFORE_TRAIN = 100
EVAL_EPISODES = 1
EVAL_INTERVAL = 5

MAX_STEPS = 500  # max steps to explore the environment
EVAL_MAX_STEPS = 100
TARGET_UPDATE_INTERVAL = 10  # target net's update interval when using hard update
TARGET_TAU = 0.001  # target net's soft update parameter

N_AGENTS = 5
N_USERS = 3 * N_AGENTS  # reasonable density
N_REQUESTS = 10  # time slot 1h, 10 tasks/s
AGENT_CAPACITY = 1.7  # content size mean = 1 default=8
AGENT_VIEW = N_AGENTS - 1
TEMPERATURE = 0.1
REPLACEMENT_FACTOR = 1
ACCURACY_FACTOR = 1

MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ACTOR_HIDDEN_SIZE = 64
CRITIC_HIDDEN_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
ENTROPY_REG = 0.08

REWARD_DISCOUNTED_GAMMA = 0.95

EPSILON_START = 1.
EPSILON_END = 0.
EPSILON_DECAY = MAX_EPISODES / 4.

DONE_PENALTY = None

RANDOM_SEED = 2023


def run(env_id: str, algo_id: str, algo_handle: Type[Agent], actor_output_act=None, critic_output_act=None,
        add_random=True):
    env = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY, agent_view=AGENT_VIEW,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE, eta=REPLACEMENT_FACTOR,
                            beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY,
                                 n_requests=N_REQUESTS, max_steps=EVAL_MAX_STEPS, temperature=TEMPERATURE,
                                 eta=REPLACEMENT_FACTOR, beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    state_dim = env.n_agents * env.n_models * 2
    action_dim = env.n_actions
    if actor_output_act is None:
        actor_output_act = action_values_softmax
    if critic_output_act is None:
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
                       epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                       target_update_step=TARGET_UPDATE_INTERVAL)
    algo_random = Base(env=env, state_dim=state_dim, action_dim=action_dim,
                       device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
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
    episodes = []
    eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
    eval_rewards_c_mean_sum = []
    eval_rewards_s_mean_sum = []
    eval_rewards_mean = []  # average cumulative reward of each agent in each episode
    eval_rewards_std = []
    eval_rewards_c_mean = []
    eval_rewards_s_mean = []
    eval_infos = []
    eval_cache_stats = []
    eval_times = []
    train_times = []

    random_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
    random_rewards_c_mean_sum = []
    random_rewards_s_mean_sum = []
    random_rewards_mean = []  # average cumulative reward of each agent in each episode
    random_rewards_std = []
    random_rewards_c_mean = []
    random_rewards_s_mean = []
    random_infos = []

    print(f"env_id: {env_id}, algo_id: {algo_id}")
    print(f"n_agents: {N_AGENTS}, n_users: {N_USERS}, n_requests: {N_REQUESTS}, "
          f"agent_capacity:{AGENT_CAPACITY}, agent_view:{AGENT_VIEW}")
    print(f"legal caching actions: ({env.n_actions} in total)")

    interact_start_time = start_time = time.time()
    # cnt = 0
    while algo.n_episodes < MAX_EPISODES:
        # print(f"{algo.n_episodes}, {cnt}")
        # cnt += 1
        algo.interact()
        if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
            # print('train')
            algo.train()

        if algo.episode_done and ((algo.n_episodes + 1) % EVAL_INTERVAL == 0):
            episode_seed = random.randint(0, 2023)
            interact_end_time = time.time()
            rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env, EVAL_EPISODES,
                                                                                      episode_seed)
            eval_end_time = time.time()
            n_rewards_mu, n_rewards_std = agg_double_list(rewards)
            n_rewards_c_mu = mean_mean_list(rewards_c)
            n_rewards_s_mu = mean_mean_list(rewards_s)
            rewards_mu_sum = np.sum(n_rewards_mu)
            reward_c_mu_sum = np.sum(n_rewards_c_mu)
            reward_s_mu_mean = np.mean(n_rewards_s_mu)
            train_time = interact_end_time - interact_start_time
            eval_time = eval_end_time - interact_end_time
            print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                  % (algo.n_episodes + 1, rewards_mu_sum, train_time, eval_time))
            for info in infos:
                print(info)
            for u_info in users_info:
                print(u_info)
            eval_rewards_mean_sum.append(rewards_mu_sum)
            eval_rewards_c_mean_sum.append(reward_c_mu_sum)
            eval_rewards_s_mean_sum.append(reward_s_mu_mean)
            eval_rewards_mean.append(n_rewards_mu)
            eval_rewards_std.append(n_rewards_std)
            eval_rewards_c_mean.append(n_rewards_c_mu)
            eval_rewards_s_mean.append(n_rewards_s_mu)
            eval_infos.append(infos)
            episodes.append(algo.n_episodes + 1)
            eval_times.append(eval_time)
            train_times.append(train_time)

            # random algorithm as the baseline
            if add_random:
                rewards, infos, rewards_c, rewards_s, _, _ = algo_random.evaluation(env, EVAL_EPISODES, episode_seed)
                n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                n_rewards_c_mu = mean_mean_list(rewards_c)
                n_rewards_s_mu = mean_mean_list(rewards_s)
                rewards_mu_sum = np.sum(n_rewards_mu)
                reward_c_mu_sum = np.sum(n_rewards_c_mu)
                reward_s_mu_mean = np.mean(n_rewards_s_mu)

                random_rewards_mean_sum.append(rewards_mu_sum)
                random_rewards_c_mean_sum.append(reward_c_mu_sum)
                random_rewards_s_mean_sum.append(reward_s_mu_mean)
                random_rewards_mean.append(n_rewards_mu)
                random_rewards_std.append(n_rewards_std)
                random_rewards_c_mean.append(n_rewards_c_mu)
                random_rewards_s_mean.append(n_rewards_s_mu)
                random_infos.append(infos)

            interact_start_time = time.time()
    end_time = time.time()
    total_time = end_time - start_time

    eval_rewards_mean = np.array(eval_rewards_mean)
    eval_rewards_std = np.array(eval_rewards_std)
    eval_rewards_c_mean = np.array(eval_rewards_c_mean)
    eval_rewards_s_mean = np.array(eval_rewards_s_mean)

    if add_random:
        random_rewards_mean = np.array(random_rewards_mean)
        random_rewards_std = np.array(random_rewards_std)
        random_rewards_c_mean = np.array(random_rewards_c_mean)
        random_rewards_s_mean = np.array(random_rewards_s_mean)

    time_str = time.strftime('%m_%d_%H_%M')

    # draw the locations of users and edges
    draw_loc(env, time_str, algo_id)

    # show the popularity of each model for caching in the entire system
    # plot_cache_popularity(env.world.models, eval_cache_stats, time_str, algo_id)

    # save rewards information as a table
    sorted_eval_rewards = np.array([[eval_rewards_mean[:, i], eval_rewards_c_mean[:, i], eval_rewards_s_mean[:, i]]
                                    for i in range(N_AGENTS)]).reshape(-1, len(episodes)).T
    episodes_rewards = np.column_stack((episodes, train_times, eval_times, sorted_eval_rewards))
    cols = ['Episodes', 'Train Time', 'Evaluation Time']
    for i in range(N_AGENTS):
        cols.append("Agent %d Rewards" % i)
        cols.append("%d Cache Hit" % i)
        cols.append("%d Switch Cost" % i)
    df = pd.DataFrame(episodes_rewards, columns=cols)
    df.to_csv("./output/%s_%s_data.csv" % (time_str, algo_id), index=False, float_format='%.4f')

    # save all parameters of the environment and algorithm
    save_parameters(algo_id, time_str, total_time)

    # plot the average reward versus episode
    plt.style.use('seaborn-v0_8-whitegrid')
    palette_0, palette_1, palette_2 = plt.get_cmap('Set1'), plt.get_cmap('Set2'), plt.get_cmap('Set1')
    fig, axs = plt.subplots(2, 2, figsize=(23, 9))

    axs[0, 0].plot(episodes, eval_rewards_mean_sum, color='r', label=algo_id)
    axs[0, 0].plot(episodes, random_rewards_mean_sum, color='b', label='random')
    axs[0, 0].set_title("Sum of Episode Average Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend(loc='upper right', shadow=True)

    for i in range(N_AGENTS):
        label = f"Agent {i}"
        color = palette_0(i)
        axs[0, 1].plot(episodes, eval_rewards_mean[:, i], color=color, label=label)
        sup = list(map(lambda x, y: x + y, eval_rewards_mean[:, i], eval_rewards_std[:, i]))
        inf = list(map(lambda x, y: x - y, eval_rewards_mean[:, i], eval_rewards_std[:, i]))
        axs[0, 1].fill_between(episodes, inf, sup, color=color, alpha=0.2)
    axs[0, 1].set_title("Episode Rewards of Agents")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].legend(loc='upper right', shadow=True)

    color = palette_1(0)
    axs[1, 0].plot(episodes, eval_rewards_c_mean_sum, color=color, label=algo_id)
    color = palette_1(1)
    axs[1, 0].plot(episodes, random_rewards_c_mean_sum, color=color, label='random')
    axs[1, 0].set_title("Episode Average Cache Hit")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Cache Hit Reward")
    axs[1, 0].legend(loc='upper right', shadow=True)

    color = palette_2(0)
    axs[1, 1].plot(episodes, eval_rewards_s_mean_sum, color=color, label=algo_id)
    color = palette_2(1)
    axs[1, 1].plot(episodes, random_rewards_s_mean_sum, color=color, label='random')
    axs[1, 1].set_title("Episode Average Switch Cost")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Switch Cost")
    axs[1, 1].legend(loc='upper right', shadow=True)

    fig.tight_layout()
    plt.savefig("./output/%s_%s_reward.png" % (time_str, algo_id))


#  run MARL algorithm with different settings of hyperparameters
def run_params(algo_id: str, algo_handle: Type[Agent]):
    n_agents = 5
    n_users = 3 * n_agents
    env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=AGENT_CAPACITY, agent_view=2,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE, eta=REPLACEMENT_FACTOR,
                            beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    state_dim = env.n_agents * env.n_models * 2
    action_dim = env.n_actions
    actor_output_act = action_values_softmax
    critic_output_act = identity

    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [64, 128, 256]
    rewards_sum = []
    n_episode = 1000
    episodes = np.arange(EVAL_INTERVAL, n_episode + 1, EVAL_INTERVAL)

    start_time = time.time()

    rewards_sum.append([])
    for bs in batch_sizes:
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=bs,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL)
        eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode

        interact_start_time = time.time()
        while algo.n_episodes < n_episode:
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

        rewards_sum[0].append(eval_rewards_mean_sum)

    rewards_sum.append([])
    for lr in learning_rates:
        algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
                           device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
                           reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
                           actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
                           actor_output_act=actor_output_act, critic_output_act=critic_output_act,
                           critic_loss=CRITIC_LOSS, actor_lr=lr, critic_lr=lr,
                           optimizer_type="adam", entropy_reg=ENTROPY_REG,
                           max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                           episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
                           epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
                           target_update_step=TARGET_UPDATE_INTERVAL)
        eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode

        interact_start_time = time.time()
        while algo.n_episodes < n_episode:
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

        rewards_sum[1].append(eval_rewards_mean_sum)

    # rewards_sum.append([])
    # for lr in learning_rates:
    #     algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
    #                        device=DEVICE, memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS,
    #                        reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
    #                        actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
    #                        actor_output_act=actor_output_act, critic_output_act=critic_output_act,
    #                        critic_loss=CRITIC_LOSS, actor_lr=ACTOR_LR, critic_lr=lr,
    #                        optimizer_type="adam", entropy_reg=ENTROPY_REG,
    #                        max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
    #                        episodes_before_train=EPISODES_BEFORE_TRAIN, epsilon_start=EPSILON_START,
    #                        epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_tau=TARGET_TAU,
    #                        target_update_step=TARGET_UPDATE_INTERVAL)
    #     eval_rewards_mean_sum = []  # the sum of rewards of all agents in each episode
    #
    #     interact_start_time = time.time()
    #     while algo.n_episodes < n_episode:
    #         algo.interact()
    #         if algo.n_episodes >= 20:
    #             algo.train()
    #
    #         if algo.episode_done and ((algo.n_episodes + 1) % EVAL_INTERVAL == 0):
    #             interact_end_time = time.time()
    #             rewards, infos, rewards_c, rewards_s, _, _ = algo.evaluation(env, EVAL_EPISODES)
    #             eval_end_time = time.time()
    #             n_rewards_mu, n_rewards_std = agg_double_list(rewards)
    #             rewards_mu_sum = np.sum(n_rewards_mu)
    #             train_time = interact_end_time - interact_start_time
    #             eval_time = eval_end_time - interact_end_time
    #             print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
    #                   % (algo.n_episodes + 1, rewards_mu_sum, train_time, eval_time))
    #
    #             eval_rewards_mean_sum.append(rewards_mu_sum)
    #             interact_start_time = time.time()
    #
    #     rewards_sum[2].append(eval_rewards_mean_sum)

    end_time = time.time()
    total_time = end_time - start_time

    time_str = time.strftime('%m_%d_%H_%M')
    plot_rewards_sum(episodes, rewards_sum, time_str, algo_id, learning_rates, batch_sizes)
    save_parameters(algo_id, time_str, total_time, 'param')


# run MARL algorithm with different zipf parameters for the request distribution
# compare the numbers of cached models
def run_cache(algo_id: str, algo_handle: Type[Agent]):
    env = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_users=N_USERS, agent_capacity=AGENT_CAPACITY, agent_view=AGENT_VIEW,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE, eta=REPLACEMENT_FACTOR,
                            beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    state_dim = env.n_agents * env.n_models * 2
    action_dim = env.n_actions
    actor_output_act = action_values_softmax
    critic_output_act = identity

    zipf_parameters = [0.5, 1., 2.]
    eval_cache_stats = []  # numbers of models caching in the last episode

    start_time = time.time()
    for i, param in enumerate(zipf_parameters):
        env.world.zipf_param = param
        env.reset()
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
            if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
                algo.train()

            if algo.episode_done and ((algo.n_episodes + 1) % EVAL_INTERVAL == 0):
                episode_seed = random.randint(0, 2023)
                interact_end_time = time.time()
                rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env, 1, episode_seed)
                eval_end_time = time.time()
                n_rewards_mu, n_rewards_std = agg_double_list(rewards)
                rewards_mu_sum = np.sum(n_rewards_mu)
                train_time = interact_end_time - interact_start_time
                eval_time = eval_end_time - interact_end_time
                print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                      % (algo.n_episodes + 1, rewards_mu_sum, train_time, eval_time))
                for info in infos:
                    print(info)
                if algo.n_episodes + 1 == MAX_EPISODES:
                    cache_stats = np.mean(stats['cache'], axis=0)

                interact_start_time = time.time()

        eval_cache_stats.append(cache_stats)

    total_time = time.time() - start_time
    time_str = time.strftime('%m_%d_%H_%M')
    plot_cache_popularity(env.world.models, eval_cache_stats, zipf_parameters, time_str, algo_id)
    save_parameters(algo_id, time_str, total_time, 'cache')


def run_delay(algo_id: str, algo_handle: Type[Agent]):
    max_episodes = 20
    n_agents = 3
    n_users = 3 * n_agents
    env = EdgeMultiAgentEnv(n_agents=n_agents, n_users=n_users, agent_capacity=1.65, agent_view=AGENT_VIEW,
                            n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE, eta=REPLACEMENT_FACTOR,
                            beta=ACCURACY_FACTOR, seed=RANDOM_SEED)
    state_dim = env.n_agents * env.n_models * 2
    action_dim = env.n_actions
    actor_output_act = action_values_softmax
    critic_output_act = identity
    # print(env.n_actions)
    # print(env.world.model_sizes)
    # print(env.num2action)

    average_delays = []  # average delays of the requests of every type of model in some episodes
    cache_hit_ratio = []  # overall cache hit ratio
    episodes = []

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

    interact_start_time = start_time = time.time()
    while algo.n_episodes < max_episodes:
        algo.interact()
        if algo.n_episodes >= EPISODES_BEFORE_TRAIN:
            algo.train()

        if algo.episode_done and ((algo.n_episodes + 1) % EVAL_INTERVAL == 0):
            episode_seed = random.randint(0, 2023)
            interact_end_time = time.time()
            rewards, infos, rewards_c, rewards_s, users_info, stats = algo.evaluation(env, 1, episode_seed)
            eval_end_time = time.time()
            n_rewards_mu, n_rewards_std = agg_double_list(rewards)
            rewards_mu_sum = np.sum(n_rewards_mu)
            train_time = interact_end_time - interact_start_time
            eval_time = eval_end_time - interact_end_time
            print("Episode %d, Total Reward %.2f, Interaction & Train Time %.2f s, Evaluation Time %.2f s"
                  % (algo.n_episodes + 1, rewards_mu_sum, train_time, eval_time))
            for info in infos:
                print(info)
            # for u_info in users_info:
            #     print(u_info)
            delay = np.mean(stats['delay'], axis=0)
            average_delays.append(delay)
            hit_ratio = np.mean(stats['ratio'], axis=0)
            cache_hit_ratio.append(hit_ratio)
            episodes.append(algo.n_episodes + 1)
            interact_start_time = time.time()

    total_time = time.time() - start_time
    time_str = time.strftime('%m_%d_%H_%M')
    plot_delay(average_delays, cache_hit_ratio, episodes, env.world.models, time_str, algo_id)
    save_parameters(algo_id, time_str, total_time, 'delay')


# draw the locations of users and edges in the system
def draw_loc(env: EdgeMultiAgentEnv, time_str: str, algo_id: str):
    plt.style.use('seaborn-v0_8-white')
    plt.figure(dpi=300)
    plt.grid(linestyle='--', linewidth=0.5)
    # plt.tick_params(axis='both', which='major', labelsize=14)

    colors = np.linspace(0, 1, env.n_users)
    plt.scatter(env.users_loc[:, 0], env.users_loc[:, 1], c=colors, marker='.', cmap='winter', s=40)

    colors = np.linspace(0, 0.8, env.n_agents)
    c = plt.cm.spring(colors)
    plt.scatter(env.agents_loc[:, 0], env.agents_loc[:, 1], edgecolors=c, marker='*', facecolors='none', s=80)

    legend_elements = [
        Line2D([0], [0], color=plt.cm.winter(0), marker='.', label='IoTD', linestyle='None', markersize=5),
        Line2D([0], [0], color=plt.cm.spring(0), marker='*', markerfacecolor='none', label='Edge', linestyle='None',
               markersize=9)]
    plt.legend(handles=legend_elements, frameon=True, loc='best')

    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.xlim([-50, 1050])
    plt.ylim([-50, 1050])
    plt.title('Locations of Edges and IoTDs')

    plt.savefig("./output/%s_%s_loc.png" % (time_str, algo_id))


def plot_rewards_sum(episodes: np.ndarray, rewards_sum: list, time_str: str, algo_id: str,
                     learning_rates=None, batch_sizes=None):
    plt.style.use('seaborn-v0_8-white')

    if learning_rates is not None and batch_sizes is not None:
        n = len(learning_rates)
        m = len(batch_sizes)
        fig, axes = plt.subplots(2, 1)

        colors = plt.cm.hsv(np.linspace(0, 0.8, m))
        for j, bs in enumerate(batch_sizes):
            label = f"batch size={bs}"
            axes[0].plot(episodes, rewards_sum[0][j], color=colors[j], label=label)
        # axes[0].set_title(f"learning rate={learning_rates[i]: 0e}")
        axes[0].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
        axes[0].set_ylabel(r"$\sum_m R_m^t$")
        axes[0].set_xlabel('Episode')
        axes[0].grid(linestyle='--', linewidth=0.5)

        colors = plt.cm.rainbow(np.linspace(0, 0.8, n))
        for j, lr in enumerate(learning_rates):
            label = f"learning rate={lr:.2e}"
            axes[1].plot(episodes, rewards_sum[1][j], color=colors[j], label=label)
        axes[1].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
        axes[1].set_ylabel(r"$\sum_m R_m^t$")
        axes[1].set_xlabel('Episode')
        axes[1].grid(linestyle='--', linewidth=0.5)

        # colors = plt.cm.turbo(np.linspace(0, 0.8, n))
        # for j, lr in enumerate(learning_rates):
        #     label = f"critic learning rate={lr}"
        #     axes[2].plot(episodes, rewards_sum[1][j], color=colors[j], label=label)
        # axes[2].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
        # axes[2].set_ylabel(r"$\sum_m R_m^t$")
        # axes[2].set_xlabel('Episode')
        # axes[2].grid(linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig("./output/%s_%s_rewards_param.png" % (time_str, algo_id), bbox_inches='tight', dpi=600)


def plot_cache_popularity(models: list, stats: list, params: list, time_str: str, algo_id: str):
    n_models = len(models)
    plt.style.use('seaborn-v0_8-white')
    n = len(stats)
    fig, axes = plt.subplots(n, 1)
    max_model_nums = max(len(model_types) for model_types in models)

    for i, ax in enumerate(axes.flatten()):
        ax.grid(linestyle='--', linewidth=0.5, axis='y')
        data = np.zeros((n_models, max_model_nums))
        idx = 0
        for j, model_types in enumerate(models):
            num = len(model_types)
            data[j][:num] = stats[i][idx:idx + num]
            idx += num

        colors = plt.cm.cool(np.linspace(0, 1, max_model_nums))
        index = [models[0].type for models in models]
        bar_width = 0.4
        y_offset = np.zeros(n_models)

        for row in range(max_model_nums):
            ax.bar(x=index, height=data[:, row], width=bar_width, bottom=y_offset, color=colors[row])
            y_offset += data[:, row]

        ax.set_ylabel('Number of caches')
        ax.set_title(f"Zipf Parameters = {params[i]: .1f}")
        ax.yaxis.grid(linestyle='--', linewidth=0.5)

    # plt.xlabel('Model Type')
    legends = ["Accuracy {:d}".format(i) for i in range(max_model_nums)]
    ax = axes[n - 1]
    ax.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              bbox_transform=fig.transFigure, ncol=4, frameon=True)

    plt.tight_layout()
    plt.savefig("./output/%s_%s_cache.png" % (time_str, algo_id), bbox_inches='tight', dpi=600)


def plot_delay(delays: list, cache_hit_ratio: list, episodes: list, models: list, time_str: str, algo_id: str):
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(2, 1)

    colors = plt.cm.rainbow(np.linspace(0, 0.6, 2))

    ax = axes[0]
    ax.plot(episodes, delays, color=colors[0])
    ax.set_ylabel('Average Delay (ms)')
    ax.set_xlabel('Episode')
    ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax = axes[1]
    ax.plot(episodes, cache_hit_ratio, color=colors[1])
    ax.set_ylabel('Average Cache Hit Ratio')
    ax.set_xlabel('Episode')
    ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.savefig("./output/%s_%s_delay.png" % (time_str, algo_id), bbox_inches='tight', dpi=600)


def save_parameters(algo_id: str, time_str: str, total_time: float, method: str):
    parameters = f"algo_id: {algo_id}, random seed: {RANDOM_SEED},\n" \
                 f"n_agents: {N_AGENTS},\nn_users: {N_USERS},\n" \
                 f"n_requests: {N_REQUESTS},\n" \
                 f"agent_capacity: {AGENT_CAPACITY},\nagent_view: {AGENT_VIEW},\n" \
                 f"replacement factor: {REPLACEMENT_FACTOR}, accuracy factor: {ACCURACY_FACTOR}\n" \
                 f"actor_hidden_size: {ACTOR_HIDDEN_SIZE}, critic_hidden_size: {CRITIC_HIDDEN_SIZE},\n" \
                 f"batch size: {BATCH_SIZE}, critic loss: {CRITIC_LOSS},\n" \
                 f"episodes number: {MAX_EPISODES}, episodes before training: {EPISODES_BEFORE_TRAIN},\n" \
                 f"evaluation interval: {EVAL_INTERVAL}, evaluation episodes number: {EVAL_EPISODES},\n" \
                 f"max steps: {MAX_STEPS}, evaluation_max_steps: {EVAL_MAX_STEPS},\n" \
                 f"temperature: {TEMPERATURE},\n" \
                 f"total time: {total_time: .4f} s"
    f = open(f"./output/{time_str}_{algo_id}_{method}.txt", "w")
    f.write(parameters)
    f.close()
