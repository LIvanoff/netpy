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
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] < 0:
                    x[i][j] = 0
        return x


class ELU(object):
    def __call__(self, x: torch.Tensor, alpha: float = 1.):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] < 0:
                    x[i][j] = alpha * (torch.exp(x[i][j]) - 1)
        return x


class LeakyReLU(object):
    def __call__(self, x: torch.Tensor, alpha: float = 0.01):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] < 0:
                    x[i][j] = alpha * x[i][j]
        return x


class sign(object):
    def __call__(self, x: torch.Tensor):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] > 0:
                    x[i][j] = 1
                elif x[i][j] == 0:
                    x[i][j] = 0
                else:
                    x[i][j] = -1
        return x


class Heaviside(object):
    def __call__(self, x: torch.Tensor, value: float):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] > 0:
                    x[i][j] = 1
                elif x[i][j] == 0:
                    x[i][j] = value
                else:
                    x[i][j] = -1
        return x


class Hardshrink(object):
    def __call__(self, x: torch.Tensor, lambda_: float = 0.5):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] <= lambda_ or x[i][j] >= -lambda_:
                    x[i][j] = 0
        return x


class Hardsigmoid(object):
    def __call__(self, x: torch.Tensor):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                if x[i][j] >= 3:
                    x[i][j] = 1
                elif x[i][j] <= -3:
                    x[i][j] = 0
                else:
                    x[i][j] = x[i][j] / 6. + 0.5
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
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                x[i][j] = 1. if x[i][j] > threshold else value
        return x
