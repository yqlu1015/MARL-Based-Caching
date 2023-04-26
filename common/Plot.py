import time
from typing import Type, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from common.environment.Environment import EdgeMultiAgentEnv
from common.utils import sim


def plot_from_data(file_names: list, algo_ids: list, suffix='comp', n_agents=5):
    episodes = []
    rewards_sum = []
    delays = []
    cache_hit_ratios = []
    switch = []
    qoe = []
    agent_qoes_mu = []
    agent_qoes_std = []
    agent_switches_mu = []
    agent_switches_std = []
    for i, file_name in enumerate(file_names):
        df = pd.read_csv(file_name)
        episodes.append(df['episode'].to_numpy())
        rewards_sum.append(df['reward sum'].to_numpy())
        delays.append(df['average delay'].to_numpy())
        cache_hit_ratios.append(df['cache hit ratio'].to_numpy())
        switch.append(df['average switch cost'].to_numpy())
        qoe.append(df['average qoe'].to_numpy())
        agent_qoes_mu.append([])
        agent_qoes_std.append([])
        agent_switches_mu.append([])
        agent_switches_std.append([])
        for j in range(n_agents):
            agent_qoes_mu[i].append(df[f"{j}_qoe_mu"].to_numpy())
            agent_qoes_std[i].append(df[f"{j}_qoe_std"].to_numpy())
            agent_switches_mu[i].append(df[f"{j}_switch_mu"].to_numpy())
            agent_switches_std[i].append(df[f"{j}_switch_std"].to_numpy())

    # rewards_sum = np.array(rewards_sum)
    # delays = np.array(delays)
    # cache_hit_ratios = np.array(cache_hit_ratios)
    args = {'qoe_mu': agent_qoes_mu, 'qoe_std': agent_qoes_std,
            'switch_mu': agent_switches_mu, 'switch_std': agent_switches_std}
    time_str = time.strftime('%m_%d_%H_%M')
    plot_rewards(episodes=episodes[0], rewards_sum=rewards_sum, time_str=time_str, algo_ids=algo_ids, suffix=suffix)
    plot_delay(delays=delays, cache_hit_ratio=cache_hit_ratios, episodes=episodes[0], time_str=time_str,
               algo_ids=algo_ids, switch=switch, qoe=qoe, **args)


# plot rewards with different settings of hyperparameters
# one csv file contains multiple columns of rewards
def plot_rewards_from_files(file_names: list, algo_id: str, learning_rates: list, batch_sizes: list):
    rewards_sum = []
    episodes = []
    n = len(file_names)

    for i, file_name in enumerate(file_names):
        rewards_sum.append([])
        df = pd.read_csv(file_name)
        episodes = df.iloc[:, 0].to_numpy()
        for j in range(1, len(df.columns)):
            rewards_sum[i].append(df.iloc[:, j].to_numpy())

    time_str = time.strftime('%m_%d_%H_%M')
    plot_rewards(episodes=episodes, rewards_sum=rewards_sum, time_str=time_str, algo_id=algo_id,
                 learning_rates=learning_rates, batch_sizes=batch_sizes, suffix='param', n_plots=n)


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

    plt.savefig(f"./output/{time_str}_{algo_id}_loc.png")


# plot the similarity between any two agents' cache vs. their physical distance
def plot_sim_dis(env: EdgeMultiAgentEnv, time_str: str, algo_id: str):
    distance = []
    similarity = []
    num = env.n_agents * (env.n_agents - 1)
    colors = np.linspace(0, 1, num)
    for i, agent in enumerate(env.world.agents):
        for j, agent_j in enumerate(env.world.agents):
            if j == i:
                continue
            dis = np.linalg.norm(agent.location - agent_j.location)
            simi = sim(env.world.observations[i].cache, env.world.observations[j].cache)
            distance.append(dis)
            similarity.append(simi)

    plt.scatter(distance, similarity, c=colors, marker='x', cmap='rainbow')
    plt.xlabel('Distance (m)')
    plt.ylabel('Similarity')
    plt.grid(linestyle='--', linewidth=0.5)

    plt.savefig(f"./output/{time_str}_{algo_id}_similarity.png", bbox_inches='tight', dpi=300)


def plot_rewards(episodes: np.ndarray, rewards_sum: list, time_str: str, algo_id: str = None, algo_ids: list = None,
                 learning_rates=None, batch_sizes=None, suffix='comp', n_plots=1, rewards: list = None):
    # plt.style.use('seaborn-v0_8-white')
    if learning_rates is not None and batch_sizes is not None:
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
        colors = ["#ff595e", "#ffca3a", "#1982c4", "#8ac926", "#6a4c93"]

        ax = axes if n_plots == 1 else axes[0]
        for j, bs in enumerate(batch_sizes):
            label = f"batch size={bs}"
            ax.plot(episodes, rewards_sum[0][j], color=colors[j], linewidth=1, label=label)
        # axes[0].set_title(f"learning rate={learning_rates[i]: 0e}")
        # axes[0].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
        ax.legend(frameon=True)
        ax.set_ylabel(r"$\sum_m R_m^t$")
        ax.set_xlabel('Episode')
        ax.grid(linestyle='--', linewidth=0.5)

        if n_plots >= 2:
            critic_ids = ['mfac', 'ac', 'ddpg']
            for j, lr in enumerate(learning_rates):
                label = f"critic learning rate={lr:.2e}" if algo_id in critic_ids else f"learning rate={lr:.2e}"
                axes[1].plot(episodes, rewards_sum[1][j], color=colors[j], linewidth=1, label=label)
            # axes[1].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
            axes[1].legend(frameon=True)
            axes[1].set_ylabel(r"$\sum_m R_m^t$")
            axes[1].set_xlabel('Episode')
            axes[1].grid(linestyle='--', linewidth=0.5)

        if n_plots >= 3:
            for j, lr in enumerate(learning_rates):
                label = f"actor learning rate={lr: .2e}"
                axes[2].plot(episodes, rewards_sum[2][j], color=colors[j], linewidth=1, label=label)
            # axes[2].legend(loc='lower left', bbox_to_anchor=(1.04, 0), borderaxespad=0, frameon=True)
            axes[2].legend(frameon=True)
            axes[2].set_ylabel(r"$\sum_m R_m^t$")
            axes[2].set_xlabel('Episode')
            axes[2].grid(linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"./output/{time_str}_{algo_id}_rewards_{suffix}.png", bbox_inches='tight', dpi=300)

    elif algo_ids is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        palette = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]

        for j, reward in enumerate(rewards_sum):
            label = f"{algo_ids[j]}"
            ax.plot(episodes, reward, color=palette[j], linewidth=1, label=label)
        ax.legend(frameon=True)
        ax.set_ylabel(r"$\sum_t R(t)$")
        ax.set_xlabel('Episode')
        ax.set_title('Episode Reward Sum')
        # ax.grid(linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"./output/{time_str}_rewards_{suffix}.png", bbox_inches='tight', dpi=300)

        if rewards is not None:
            plt.clf()
            n_plots = len(algo_ids)
            fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
            for i, agent_reward in enumerate(rewards):
                ax = axes[i] if n_plots > 1 else axes
                colors = plt.cm.jet(np.linspace(0.2, 0.8, len(agent_reward)))
                for j, reward in enumerate(agent_reward):
                    ax.plot(episodes, reward, color=colors[j], linewidth=1, label=f"Agent {j}")
                ax.legend(frameon=True)
                ax.set_ylabel(r"$R_m^t$")
                ax.set_xlabel('Episode')
                ax.set_title(f"Agent Reward {algo_ids[i]}")

            plt.tight_layout()
            plt.savefig(f"./output/{time_str}_agent_rewards.png", bbox_inches='tight', dpi=300)


def plot_delay(delays: list, cache_hit_ratio: list, episodes: np.ndarray, time_str: str,
               algo_ids: List[str], switch: list, qoe: list, qoe_mu: list = None, qoe_std: list = None,
               switch_mu: list = None, switch_std: list = None):
    plt.style.use('seaborn-v0_8-white')
    n = len(cache_hit_ratio)
    # delay, switch, cache hit ratio, qoe
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    palette = ['#AF87CE', '#EA1A7F', '#FEC603', '#A8F387', '#16D6FA']
    palette2 = ['#00ECC2', '#0078FF', '#FFD75F', '#FF8A25', '#FF4359']
    palette3 = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    ax = axes[0, 0]
    for i in range(n):
        ax.plot(episodes, qoe[i], color=palette2[i % len(palette2)], linewidth=1,
                label=algo_ids[i])
    ax.legend(frameon=True)
    ax.set_ylabel('QoE')
    ax.set_xlabel('Episode')
    # ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title("Average QoE")

    ax = axes[0, 1]
    for i in range(n):
        ax.plot(episodes, switch[i], color=palette2[i % len(palette2)], linewidth=1,
                label=algo_ids[i])
    ax.legend(frameon=True)
    ax.set_ylabel('Switch Cost (MB)')
    ax.set_xlabel('Episode')
    # ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title("Average Switch Cost")

    ax = axes[1, 0]
    for i in range(n):
        ax.plot(episodes, delays[i], color=palette2[i % len(palette2)], linewidth=1, label=algo_ids[i])
    ax.legend(frameon=True)
    ax.set_ylabel('Delay (ms)')
    ax.set_xlabel('Episode')
    # ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title("Average Offloading Delay")

    ax = axes[1, 1]
    for i in range(n):
        ax.plot(episodes, cache_hit_ratio[i], color=palette2[i % len(palette2)], linewidth=1,
                label=algo_ids[i])
    ax.legend(frameon=True)
    ax.set_ylabel('Cache Hit Ratio (%)')
    ax.set_xlabel('Episode')
    # ax.grid(linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title("Average Cache Hit Ratio")


    plt.tight_layout()
    plt.savefig("./output/%s_delay.png" % time_str, bbox_inches='tight', dpi=300)

    if qoe_mu is not None:
        plt.clf()
        n_plots = len(algo_ids)
        fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3.5 * n_plots))

        for i, (q_mu, q_std, s_mu, s_std) in enumerate(zip(qoe_mu, qoe_std, switch_mu, switch_std)):
            colors = plt.cm.hsv(np.linspace(0, 0.8, len(q_mu)))
            ax = axes[i, 0] if n_plots > 1 else axes[0]
            for j, (q_mu_j, q_std_j) in enumerate(zip(q_mu, q_std)):
                ax.plot(episodes, q_mu_j, color=colors[j], linewidth=1, label=f"Agent {j}")
                sup = list(map(lambda x, y: x + y, q_mu_j, q_std_j))
                inf = list(map(lambda x, y: x - y, q_mu_j, q_std_j))
                ax.fill_between(episodes, inf, sup, color=colors[j], alpha=0.2)
            ax.legend(frameon=True)
            ax.set_ylabel("QoE")
            ax.set_xlabel('Episode')
            ax.set_title(f"{algo_ids[i]} Average QoE")

            ax = axes[i, 1] if n_plots > 1 else axes[1]
            for j, (s_mu_j, s_std_j) in enumerate(zip(s_mu, s_std)):
                ax.plot(episodes, s_mu_j, color=colors[j], linewidth=1, label=f"Agent {j}")
                sup = list(map(lambda x, y: x + y, s_mu_j, s_std_j))
                inf = list(map(lambda x, y: x - y,s_mu_j, s_std_j))
                ax.fill_between(episodes, inf, sup, color=colors[j], alpha=0.2)
            ax.legend(frameon=True)
            ax.set_ylabel("Switch Cost (MB)")
            ax.set_xlabel('Episode')
            ax.set_title(f"{algo_ids[i]} Average Switch Cost")

        plt.tight_layout()
        plt.savefig(f"./output/{time_str}_agent_infos.png", bbox_inches='tight', dpi=300)


def plot_cache_popularity(models: list, stats: list, time_str: str, algo_id: str, params: list = None,
                          spaces: list = None):
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
        y_offset = np.zeros(n_models)
        width = 0.4

        for row in range(max_model_nums):
            ax.bar(x=index, height=data[:, row], bottom=y_offset, width=width, color=colors[row])
            y_offset += data[:, row]

        ax.set_ylabel('Average Number of Caches')
        if params is not None:
            ax.set_title(f"Zipf Parameters = {params[i]: .1f}")
        elif spaces is not None:
            ax.set_title(f"Edge Store Space = {spaces[i]: .1f}")
        ax.yaxis.grid(linestyle='--', linewidth=0.5)

    legends = ["Accuracy {:d}".format(i) for i in range(max_model_nums)]
    ax = axes[n - 1]
    ax.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 0.1),
              bbox_transform=fig.transFigure, ncol=4, frameon=True)

    plt.tight_layout()
    plt.savefig("./output/%s_%s_cache.png" % (time_str, algo_id), bbox_inches='tight', dpi=300)
