from algorithms.MFDQN import MFDQN
from algorithms.DQN import DQN
from algorithms.MFAC import MFAC
from algorithms.base import Base
from algorithms.AC import AC
from algorithms.DDPG import DDPG
from common.Runner import run_params, run_cache, run_comp
from common.Plot import plot_from_data

if __name__ == "__main__":
    # run_params('mfq', MFDQN)
    run_params('mfac', MFAC)
    # run_params('random', Base)

    # run_cache('random', Base)
    # run_cache('mfq', MFDQN)

    # run_delay('random', Base)
    # run_delay('mfq', MFDQN)
    # run_delay(['mfiq', 'mfq'], [MFIQ, MFDQN])

    # run_comp(['mfac', 'ddpg'], [MFAC, DDPG])
    run_comp(['iql', 'mfq', 'ac'], [DQN, MFDQN, AC])