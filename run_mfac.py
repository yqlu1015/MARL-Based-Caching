from algorithms.MFAC import MFAC
from common.Runner import run
from common.utils import identity

if __name__ == "__main__":
    run('edge', 'mfac', MFAC, critic_output_act=identity)