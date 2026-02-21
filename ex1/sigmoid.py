import numpy as np


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))
