import os, sys
import numpy as np
import pandas as pd


class Request(object):
    # request of a DNN is represented by an id and its position
    def __init__(self, id: int, loc: list) -> None:
        self.id = id
        self.loc = loc

    def __str__(self) -> str:
        return "dnn_id: {:d}, ".format(int(self.id)) + "loc: ({:.2f},{:.2f})".format(self.loc[0], self.loc[1])


# Generate a list of numbers which follow the zipf distribution
def generate_zipf(num_types, param, num_samples, order=None):
    ranks = np.arange(num_types) + 1
    pdf = 1 / np.power(ranks, param)
    pdf /= sum(pdf)

    # Draw samples
    if order is None:
        order = np.arange(num_types)
    return np.random.choice(order, num_samples, p=pdf)


# Generate the number of requests {q_nj}
def generate_requests(num_users=10, num_types=5, num_requests=10, zipf_param=0.6, orders=None) -> np.array:
    requests = np.zeros((num_users, num_types))
    if orders is None:
        orders = np.tile(np.arange(num_types), (num_users, 1))

    for i in range(num_users):
        request_indices = generate_zipf(num_types, zipf_param, num_requests, orders[i])
        values, counts = np.unique(request_indices, return_counts=True)
        for v, c in zip(values, counts):
            requests[i][v] = c

    return requests


# Generate sizes of contents following Pareto distribution
def generate_content_sizes(n=50, mean=1, mini=0.5):
    # shape parameter (a) = 2, scale parameter (x_m)
    # mean = a * x_m / (a - 1) if a > 1
    # classical Pareto distribution can be obtained by
    # samples = (np.random.pareto(alpha, n_agents) + 1) * x_m
    a = mean / (mean - mini)
    return (np.random.pareto(a, n) + 1) * mini
