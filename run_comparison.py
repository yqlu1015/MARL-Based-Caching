from algorithms.MFDQN import MFDQN
from algorithms.DQN import DQN
from algorithms.MFAC import MFAC
from algorithms.MFQS import MFQS
from algorithms.base import Base
from algorithms.AC import AC
from algorithms.DDPG import DDPG
from common.Runner import run_params, run_cache, run_comp
from common.Plot import plot_from_data, plot_rewards_from_files

if __name__ == "__main__":
    # file_names = ['./output/04_25_06_34_iql_data.csv', './output/04_25_08_41_mfq_data.csv']
    # plot_from_data(file_names, ['IQL', 'MFQ'], n_agents=5, suffix='comp')

    file_names = ['./output/04_25_20_28_random_data.csv', './output/04_26_07_41_ac_data.csv',
                  './output/04_25_08_41_mfq_data.csv', './output/04_25_06_34_iql_data.csv']
    plot_from_data(file_names, ['RANDOM', 'AC', 'MFQ', 'IQL'])

    # file_names = ['./output/04_14_17_21_mfq_data.csv', './output/04_14_11_11_mfqs_data.csv']
    # plot_from_data(file_names, ['MFQ', 'MFQS'])

    # file_names = ['./output/04_15_18_24_mfac_data.csv', './output/04_15_06_48_mfacs_data.csv']
    # plot_from_data(file_names, ['MFAC', 'MFACS'])

    # file_names = ['./output/04_13_12_12_mfq_data.csv', './output/04_13_11_10_mfac_data.csv',
    #               './output/04_14_00_36_ddpg_data.csv', './output/04_13_19_47_ac_data.csv', './output/04_13_05_59_iql_data.csv']
    # plot_from_data(file_names, ['MFQ', 'MFAC', 'MADDPG', 'MAAC', 'IQL'])

    # file_names = ['./output/04_11_12_16_mfq_change_p.csv', './output/04_11_06_02_mfq_change_np.csv']
    # plot_from_data(file_names, ['obs popularity', 'not obs popularity'])

    # file_names = ['./output/04_19_22_18_ddpg_param_bs.csv', './output/04_20_18_14_ddpg_param_lra.csv']
    # plot_rewards_from_files(file_names, algo_id='ddpg', learning_rates=[1e-3, 1e-4, 1e-5], batch_sizes=[4, 16, 64, 256])

    # file_names = ['./output/04_13_16_33_mfac_param_bs.csv', './output/04_14_09_37_mfac_param_lra.csv',
    #               './output/04_15_05_17_mfac_param_lrc.csv']
    # plot_rewards_from_files(file_names, algo_id='mfac', learning_rates=[1e-3, 1e-4, 1e-5], batch_sizes=[64, 128, 256])