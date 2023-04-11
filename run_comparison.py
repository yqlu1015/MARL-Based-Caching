from algorithms.MFDQN import MFDQN
from algorithms.DQN import DQN
from algorithms.MFAC import MFAC
from algorithms.MFIQ import MFIQ
from algorithms.base import Base
from algorithms.AC import AC
from algorithms.DDPG import DDPG
from common.Runner import run_params, run_cache, run_comp, run_comp_change
from common.Plot import plot_from_data

if __name__ == "__main__":
    # run_params('mfq', MFDQN)
    # run_params('mfac', MFAC)
    # run_params('random', Base)

    # run_cache('random', Base)
    # run_cache('mfq', MFDQN) - docker server running

    # run_delay('random', Base)
    # run_delay('mfq', MFDQN)
    # run_delay(['mfiq', 'mfq'], [MFIQ, MFDQN])

    # run_comp(['ac', 'ddpg'], [AC, DDPG]) - cse server running
    # run_comp(['mfq', 'mfac'], [MFDQN, MFAC])
    # run_comp_change('mfq', MFDQN)

    # file_names = ['./output/04_11_03_56_mfq_data.csv', './output/04_11_13_16_mfac_data.csv',
    #               './output/04_10_21_07_iql_data.csv', './output/04_11_01_13_ac_data.csv', './output/04_10_23_42_mfac_data.csv']
    # plot_from_data(file_names, ['mfq', 'mfac', 'iql', 'ac', 'mfac2'])

    file_names = ['./output/04_11_14_10_ddpg_data.csv', './output/04_11_01_13_ac_data.csv',
                  './output/04_10_23_42_mfac_data.csv', './output/04_10_13_07_mfq_data.csv', './output/04_10_06_43_iql_data.csv']
    plot_from_data(file_names, ['MADDPG', 'AC', 'MFAC', 'MFQ', 'IQL'])

    # file_names = ['./output/04_11_12_16_mfq_change_p.csv', './output/04_11_06_02_mfq_change_np.csv']
    # plot_from_data(file_names, ['obs popularity', 'not obs popularity'])
