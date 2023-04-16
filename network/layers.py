import numpy as np


class Linear(object):
    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output
        self.weight = np.random.normal(loc=0.0, scale=0.5, size=(self.n_input, self.n_output))
        self.bias = np.random.normal(loc=0.0, scale=0.5)

    def __call__(self, x):
        return np.dot(x.T, self.weight) + self.bias


class Sigmoid(object):
    def __call__(self, x):
        return 1. / (1 + np.exp(-x))


class ReLU(object):
    def __call__(self, x):
        if x >= 0:
            return x
        else:
            return 0


class ELU(object):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x):
        if x >= 1:
            return x
        else:
            return self.alpha * (np.exp(x) - 1)


class LeakyReLU(object):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x):
        if x >= 0:
            return x
        else:
            return self.alpha * x


class sign(object):
    def __call__(self, x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return -1


class Heaviside(object):
    def __call__(self, x):
        if x > 0:
            return 1
        else:
            return 0


class SoftPlus(object):
    def __call__(self, x):
        return np.log(1 + np.exp(x))


class Softmax(object):
    def __call__(self, x):
        return np.exp(x) / np.mean(np.exp(x))
