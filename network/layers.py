import torch


class Linear(object):
    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output

        with torch.no_grad():
            self.weight = torch.empty((self.n_output, self.n_input), requires_grad=True).normal_(mean=0, std=0.5)
            self.bias = torch.empty(self.n_output, requires_grad=True).normal_(mean=0, std=0.5)

    def __call__(self, x):
        return x @ self.weight.T + self.bias


class Sigmoid(object):
    def __call__(self, x):
        return 1. / (1. + torch.exp(-x))


class ReLU(object):
    def __call__(self, x):
        return x * (x > 0.)


class ELU(object):
    def __call__(self, x, alpha: float = 1.0):
        pass


class LeakyReLU(object):
    def __call__(self, x, alpha: float = 0.01):
        pass


class sign(object):
    def __call__(self, x):
        pass


class Heaviside(object):
    def __call__(self, x):
        pass


class BatchNormilize(object):
    """
    layer for normalizing the input of a neural network
    """
    def __call__(self, x):
        return x


class Hardshrink(object):
    def __call__(self, x, lambda_: float = 0.5):
        pass


class Hardsigmoid(object):
    def __call__(self, x):
        pass


class SoftPlus(object):
    def __call__(self, x):
        return torch.log(1 + torch.exp(x))


class Softmax(object):
    def __call__(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x))


class LogSigmoid(object):
    def __call__(self, x):
        return torch.log(1. / (1. + torch.exp(-x)))


class Tanh(object):
    def __call__(self, x):
        return torch.tanh(x)


class Threshold(object):
    def __call__(self, x, threshold):
        pass
