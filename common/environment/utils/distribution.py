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
def generate_zipf(num_types, param, num_samples):
    ranks = np.arange(num_types) + 1
    pdf = 1 / np.power(ranks, param)
    pdf /= sum(pdf)

    # Draw samples
    return np.random.choice(np.arange(num_types), num_samples, p=pdf)


# Generate a list of tuples which follow the poisson point process
def generate_ppp(x_low, x_high, y_low, y_high, param):
    # area = abs(x_high-x_low) * abs(y_high-y_low)
    num = np.random.poisson(param)
    x, y = np.random.uniform(x_low, x_high, num), np.random.uniform(y_low, y_high, num)
    return np.c_[x, y]


# Generate a list of requests where each request is of the form (o,x)
def generate_requests(num_types=5, num_requests=10, zipf_param=0.8, x_low=0, x_high=10, y_low=0,
                      y_high=10):
    request_indices = generate_zipf(num_types, zipf_param, num_requests)
    # request_indices = np.zeros(num_requests)

    ppp_params = [np.count_nonzero(request_indices == i) for i in range(num_types)]
    ordered_indices = np.arange(num_types)
    # each param in params is of the form [index,density]
    params = np.c_[ordered_indices, ppp_params]

    # param is of the form [index,density]
    # return a list of requests
    def generate_tuples(param):
        # positions = generate_ppp(x_low, x_high, y_low, y_high, param[1])
        x, y = np.random.uniform(x_low, x_high, param[1]), np.random.uniform(y_low, y_high, param[1])
        positions = np.c_[x, y]

        return [Request(param[0], p) for p in positions]

    return [request for requests in [generate_tuples(p) for p in params] for request in requests]
    # return generate_tuples(params[0])


# Generate sizes of contents following Pareto distribution
def generate_content_sizes(n=50, mean=1, mini=0.5):
    # shape parameter (a) = 2, scale parameter (x_m)
    # mean = a * x_m / (a - 1) if a > 1
    # classical Pareto distribution can be obtained by
    # samples = (np.random.pareto(alpha, n) + 1) * x_m
    a = mean / (mean - mini)
    return (np.random.pareto(a, n) + 1) * mini
