from algorithms.MFDQN import MFDQN
from algorithms.MFQS import MFQS
from algorithms.DQN import DQN
from algorithms.MFAC import MFAC
from algorithms.MFACQ import MFACQ
from algorithms.MFACQS import MFACQS
from algorithms.CACQ import CAC
from algorithms.base import Base
from algorithms.AC import AC
from algorithms.DDPG import DDPG
from common.Runner import run_params, run_cache, run_comp
from common.Plot import plot_from_data

if __name__ == "__main__":
    # run_params('mfq', MFDQN)
    # run_params('mfac', MFAC, True)
    # run_params('random', Base)
    # run_params('mfq', MFDQN, lrc=False, lra=False)
    # run_params('mfac', MFAC, lrc=False, lra=False)
    run_params('ac', AC, lrc=True, lra=True)


    # run_cache('random', Base)
    # run_cache('mfq', MFDQN)

    # run_delay('random', Base)
    # run_delay('mfq', MFDQN)
    # run_delay(['mfiq', 'mfq'], [MFQS, MFDQN])

    # run_comp(['random', 'ac', 'iql'], [Base, AC, DQN])
    # run_comp(['mfacqs'], [MFACQS])
    # run_comp(['mfacq'], [MFACQ])
    # run_comp(['cacq', 'mfac'], [CAC, MFAC])
    # run_comp(['mfq'], [MFDQN])

    # run_comp(['mfqs', 'mfq'], [MFQS, MFDQN])
    # run_comp(['mfacs', 'mfac'], [MFACS, MFAC])
