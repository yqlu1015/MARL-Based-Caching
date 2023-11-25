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
    file_names = ['./output/05_07_12_11_mfac_data.csv', './output/05_07_08_14_mfq_data.csv',
                  './output/05_07_17_16_ddpg_data.csv', './output/05_07_05_27_iql_data.csv']
    plot_from_data(file_names, ['MFAC', 'MFQ', 'DDPG', 'IQL'], n_agents=5, suffix='comp')

    # file_names = ['./output/05_03_03_51_mfac_data.csv', './output/05_03_16_05_mfac_data_indep.csv']
    # plot_from_data(file_names, ['Shared Reward', 'Independent Reward'], n_agents=5, suffix='indep_share')

    # file_names = ['./output/05_01_08_32_mfac_data.csv', './output/05_01_01_15_mfq_data.csv',
    #               './output/05_01_16_00_ddpg_data.csv', './output/05_01_09_27_iql_data.csv']
    # plot_from_data(file_names, ['MFAC', 'MFQ', 'DDPG', 'IQL'], n_agents=5, suffix='comp')

    # file_names = ['./output/05_04_14_27_mfac_data.csv', './output/05_04_08_53_mfq_data.csv',
    #               './output/05_04_20_16_ddpg_data.csv', './output/05_04_23_53_iql_data.csv',
    #               './output/05_03_22_10_random_data.csv']
    # plot_from_data(file_names, ['MFAC', 'MFQ', 'DDPG', 'IQL', 'RANDOM'], n_agents=10, suffix='comp')

    # file_names = ['./output/05_04_14_27_mfac_data.csv', './output/05_05_14_30_mfaci_data.csv']
    # plot_from_data(file_names, ['Shared Reward', 'Independent Reward'], n_agents=10, suffix='indep_share')

    # file_names = ['./output/05_03_03_51_mfac_data.csv', './output/05_03_07_50_ddpg_data.csv',
    #               './output/05_03_03_15_mfq_data.csv','./output/05_03_02_11_iql_data.csv',
    #               './output/05_02_16_41_random_data.csv']
    # plot_from_data(file_names, ['MFAC', 'DDPG', 'MFQ', 'IQL', 'RANDOM'])

    # file_names = ['./output/04_11_12_16_mfq_change_p.csv', './output/04_11_06_02_mfq_change_np.csv']
    # plot_from_data(file_names, ['obs popularity', 'not obs popularity'])

    # file_names = ['./output/04_19_22_18_ddpg_param_bs.csv', './output/04_20_18_14_ddpg_param_lra.csv']
    # plot_rewards_from_files(file_names, algo_id='ddpg', learning_rates=[1e-3, 1e-4, 1e-5], batch_sizes=[4, 16, 64, 256])

    # file_names = ['./output/04_27_00_02_ac_param_bs.csv', './output/04_27_06_47_ac_param_lrc.csv',
    #               './output/04_27_13_38_ac_param_lra.csv']
    # plot_rewards_from_files(file_names, algo_id='ac', learning_rates=[1e-3, 1e-4, 1e-5], batch_sizes=[4, 32, 256])