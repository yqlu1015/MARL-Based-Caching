from algorithms.MFDQN import MFDQN
from algorithms.MFAC import MFAC
from algorithms.base import Base
from common.Runner import run_params, run_cache, run_delay

if __name__ == "__main__":
    # run_params('mfq', MFDQN)
    # run_params('mfac', MFAC)
    # run_params('random', Base)

    # run_cache('random', Base)
    # run_cache('mfq', MFDQN)

    run_delay('random', Base)
    # run_delay('mfq', MFDQN)
