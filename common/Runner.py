import random

from common.utils import agg_double_list, mean_mean_list
from common.environment.Environment import EdgeMultiAgentEnv
from common.environment.utils.tool import int2binary
from algorithms.base import Base
import time
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

MAX_EPISODES = 1000
EPISODES_BEFORE_TRAIN = 20
EVAL_EPISODES = 3
EVAL_INTERVAL = 5

MAX_STEPS = 500  # max steps to explore the environment
EVAL_MAX_STEPS = 100
TARGET_UPDATE_INTERVAL = 10  # target net's update interval when using hard update
TARGET_TAU = 0.001  # target net's soft update parameter

# steps for alternatively updating policies and mean actions
ALTERNATE_UPDATE_STEPS = 1
N_AGENTS = 8
N_CONTENTS = 10  # default=50
N_REQUESTS = 6 * N_AGENTS  # default=12
AGENT_CAPACITY = 2  # content size mean = 1 default=8
AGENT_VIEW = 3
TEMPERATURE = 0.1
REPLACEMENT_FACTOR = 1

MEMORY_CAPACITY = 50000
BATCH_SIZE = 128
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ACTOR_HIDDEN_SIZE = 64
CRITIC_HIDDEN_SIZE = 64
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
ENTROPY_REG = 0.01

REWARD_DISCOUNTED_GAMMA = 0.95

EPSILON_START = 1.
EPSILON_END = 0.
EPSILON_DECAY = 20

DONE_PENALTY = None

RANDOM_SEED = 2023


def run(env_id: str, algo_id: str, algo_handle, actor_output_act=None, critic_output_act=None, add_random=True):
    env = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_contents=N_CONTENTS, agent_capacity=AGENT_CAPACITY,
                            agent_view=AGENT_VIEW, n_requests=N_REQUESTS, max_steps=MAX_STEPS, temperature=TEMPERATURE,
                            replacement_factor=REPLACEMENT_FACTOR, device=DEVICE)
    env_eval = EdgeMultiAgentEnv(n_agents=N_AGENTS, n_contents=N_CONTENTS, agent_capacity=AGENT_CAPACITY,
                                 n_requests=N_REQUESTS, max_steps=EVAL_MAX_STEPS, temperature=TEMPERATURE,
                                 replacement_factor=REPLACEMENT_FACTOR, device=DEVICE)
    state_dim = env.n * env.n_contents * 2
    action_dim = env.n_actions
    if actor_output_act is None:
        actor_output_act = env.action_values_softmax
    if critic_output_act is None:
        critic_output_act = env.action_values_mask
    algo = algo_handle(env=env, state_dim=state_dim, action_dim=action_dim,
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
    print(f"n_agents: {N_AGENTS}, n_contents: {N_CONTENTS}, n_requests: {N_REQUESTS}, " \
          f"agent_capacity:{AGENT_CAPACITY}, agent_view:{AGENT_VIEW}")
    print(f"content sizes:\n{env.world.content_sizes}")
    print(f"legal caching actions: ({len(env.legal_actions)} in total)")
    # for a in env.legal_actions:
    #     print(int2binary(a, env.n_contents))

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
            rewards, infos, rewards_c, rewards_s = algo.evaluation(env_eval, EVAL_EPISODES, episode_seed)
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
                rewards, infos, rewards_c, rewards_s = algo_random.evaluation(env_eval, EVAL_EPISODES, episode_seed)
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
    sorted_eval_rewards = np.array([[eval_rewards_mean[:, i], eval_rewards_c_mean[:, i], eval_rewards_s_mean[:, i]]
                                    for i in range(N_AGENTS)]).reshape(-1, len(episodes)).T
    episodes_rewards = np.column_stack((episodes, train_times, eval_times, sorted_eval_rewards))
    cols = ['Episodes', 'Train Time', 'Evaluation Time']
    for i in range(N_AGENTS):
        cols.append("Agent %d Rewards" % i)
        cols.append("%d Cache Hit" % i)
        cols.append("%d Switch Cost" % i)

    df = pd.DataFrame(episodes_rewards, columns=cols)
    df.to_csv("./output/%s_%s_%s.csv" % (time_str, env_id, algo_id), index=False, float_format='%.4f')

    parameters = f"env_id: {env_id}, algo_id: {algo_id}, random seed: {RANDOM_SEED},\n" \
                 f"n_agents: {N_AGENTS},\nn_contents: {N_CONTENTS},\nn_requests: {N_REQUESTS},\n" \
                 f"agent_capacity: {AGENT_CAPACITY},\nagent_view: {AGENT_VIEW},\n" \
                 f"replacement factor: {REPLACEMENT_FACTOR},\n" \
                 f"actor_hidden_size: {ACTOR_HIDDEN_SIZE}, critic_hidden_size: {CRITIC_HIDDEN_SIZE},\n" \
                 f"batch size: {BATCH_SIZE}, critic loss: {CRITIC_LOSS},\n" \
                 f"episodes number: {MAX_EPISODES}, episodes before training: {EPISODES_BEFORE_TRAIN},\n" \
                 f"evaluation interval: {EVAL_INTERVAL}, evaluation episodes number: {EVAL_EPISODES},\n" \
                 f"max steps: {MAX_STEPS}, evaluation_max_steps: {EVAL_MAX_STEPS},\n" \
                 f"alternate update steps: {ALTERNATE_UPDATE_STEPS}, temperature: {TEMPERATURE},\n" \
                 f"total time: {total_time} s"
    f = open("./output/%s_%s_%s.txt" % (time_str, env_id, algo_id), "w")
    f.write(parameters)
    f.close()

    # plot the average reward versus episode
    plt.style.use('seaborn-v0_8-whitegrid')
    palette_0, palette_1, palette_2 = plt.get_cmap('Set1'), plt.get_cmap('Set2'), plt.get_cmap('Set1')
    fig, axs = plt.subplots(2, 2, figsize=(23, 9))

    axs[0, 0].plot(episodes, eval_rewards_mean_sum, color='r', label=algo_id)
    axs[0, 0].plot(episodes, random_rewards_mean_sum, color='b', label='random')
    axs[0, 0].set_title("Sum of Episode Average Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("R")
    axs[0, 0].legend(loc='upper right', shadow=True)

    for i in range(N_AGENTS):
        label = f"Agent {i}"
        color = palette_0(i)
        axs[0, 1].plot(episodes, eval_rewards_mean[:, i], color=color, label=label)
        sup = list(map(lambda x, y: x + y, eval_rewards_mean[:, i], eval_rewards_std[:, i]))
        inf = list(map(lambda x, y: x - y, eval_rewards_mean[:, i], eval_rewards_std[:, i]))
        axs[0, 1].fill_between(episodes, inf, sup, color=color, alpha=0.2)
    axs[0, 1].set_title("Episode Average Rewards")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("R")
    axs[0, 1].legend(loc='upper right', shadow=True)

    color = palette_1(0)
    axs[1, 0].plot(episodes, eval_rewards_c_mean_sum, color=color, label=algo_id)
    color = palette_1(1)
    axs[1, 0].plot(episodes, random_rewards_c_mean_sum, color=color, label='random')
    axs[1, 0].set_title("Episode Average Cache Hit")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Weighted Cache Hit Rate")
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
    plt.savefig("./output/%s_%s_%s.png" % (time_str, env_id, algo_id))
