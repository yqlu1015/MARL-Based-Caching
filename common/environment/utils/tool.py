import random
import torch as th
import numpy as np


# convert  EdgeAgentStates array/tuple to 1d array of numbers
def states2array(states):
    states = np.array(states).flatten()
    return np.ravel([[state.popularity, state.cache] for state in states])


def actions2array(actions):
    actions = np.array(actions).flatten()
    return np.ravel([action.a for action in actions])


# convert an integer x to its binary representation with given width
def int2binary(x, width) -> np.ndarray:
    assert x < 2 ** width, "width not enough"
    bits = np.zeros(width, dtype=np.int8)
    x = int(x)
    for i in range(width):
        p = width - 1 - i
        bit = x >> p
        bits[i] = bit
        x -= 2 ** p * bit
    return bits


# convert a binary representation to integer
def binary2int(x) -> int:
    x = np.array(x, dtype='int')
    exp = 2 ** np.arange(len(x) - 1, -1, -1)
    return x.dot(exp)


# set the random seeds everywhere
def seed_everything(seed):
    # random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


# scale a list to reduce the relative difference between each two values
def normalize(x):
    x = np.array(x)
    y = (x-np.min(x)) / (np.max(x)-np.min(x))
    e_y = np.exp(y)
    res = e_y / np.sum(e_y) * len(e_y)
    return res
