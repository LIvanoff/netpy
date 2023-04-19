import torch


class Linear(object):
    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output

        with torch.no_grad():
            self.weight = torch.empty((self.n_output, self.n_input), requires_grad=True).normal_(mean=0, std=0.5)
            self.bias = torch.empty(self.n_output, requires_grad=True).normal_(mean=0, std=0.5)

    def __call__(self, x: torch.Tensor):
        return x @ self.weight.T + self.bias


class BatchNormilize(object):
    """
    layer for normalizing the input of a neural network
    """
    def __call__(self, x):
        return x


class Sigmoid(object):
    def __call__(self, x: torch.Tensor):
        return 1. / (1. + torch.exp(-x))


class ReLU(object):
    def __call__(self, x: torch.Tensor):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
        return x


class ELU(object):
    def __call__(self, x: torch.Tensor, alpha: float = 1.):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = alpha * (torch.exp(x[i]) - 1)
        return x


class LeakyReLU(object):
    def __call__(self, x: torch.Tensor, alpha: float = 0.01):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = alpha * x[i]
        return x


class sign(object):
    def __call__(self, x: torch.Tensor):
        for i in range(len(x)):
            if x[i] > 0:
                x[i] = 1
            elif x[i] == 0:
                x[i] = 0
            else:
                x[i] = -1
        return x


class Heaviside(object):
    def __call__(self, x: torch.Tensor, value: float):
        for i in range(len(x)):
            if x[i] > 0:
                x[i] = 1
            elif x[i] == 0:
                x[i] = value
            else:
                x[i] = -1
        return x


class Hardshrink(object):
    def __call__(self, x: torch.Tensor, lambda_: float = 0.5):
        for i in range(len(x)):
            if x[i] <= lambda_ or x[i] >= -lambda_:
                x[i] = 0
        return x


class Hardsigmoid(object):
    def __call__(self, x: torch.Tensor):
        for i in range(len(x)):
            if x[i] >= 3:
                x[i] = 1
            elif x[i] <= -3:
                x[i] = 0
            else:
                x[i] = x[i] / 6. + 0.5
        return x


class SoftPlus(object):
    def __call__(self, x: torch.Tensor):
        return torch.log(1 + torch.exp(x))


class Softmax(object):
    def __call__(self, x: torch.Tensor):
        return torch.exp(x) / torch.sum(torch.exp(x))


class LogSigmoid(object):
    def __call__(self, x: torch.Tensor):
        return torch.log(1. / (1. + torch.exp(-x)))


class Tanh(object):
    def __call__(self, x: torch.Tensor):
        return torch.tanh(x)


class Threshold(object):
    def __call__(self, x: torch.Tensor, threshold: float, value: float = 0.):
        for i in range(len(x)):
            x[i] = 1. if x[i] > threshold else value
        return x
