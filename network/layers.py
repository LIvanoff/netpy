import torch


class Linear(object):
    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output

        with torch.no_grad():
            self.weight = torch.empty((self.n_output, self.n_input), requires_grad=True).normal_(mean=0, std=0.5)
            self.bias = torch.empty(self.n_output, requires_grad=True).normal_(mean=0, std=0.5)

        # self.weight = np.random.normal(loc=0.0, scale=0.5, size=(self.n_input, self.n_output))
        # self.bias = np.random.normal(loc=0.0, scale=0.5, size=self.n_output)

    def __call__(self, x):
        return x @ self.weight.T + self.bias  # np.dot(x.T, self.weight) + self.bias


class Sigmoid(object):
    def __call__(self, x):
        return 1. / (1. + torch.exp(-x))


class ReLU(object):
    def __call__(self, x):
        if x >= 0:
            return x
        else:
            return 0


class ELU(object):
    def __call__(self, x, alpha: float = 1.0):
        if x >= 1:
            return x
        else:
            return alpha * (torch.exp(x) - 1)


class LeakyReLU(object):
    def __call__(self, x, alpha: float = 0.01):
        if x >= 0:
            return x
        else:
            return alpha * x


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


class Hardshrink(object):
    def __call__(self, x, lambda_: float = 0.5):
        if x > lambda_:
            return x
        elif x < -lambda_:
            return x
        else:
            return 0


class Hardsigmoid(object):
    def __call__(self, x):
        if x >= 3:
            return 1
        elif x <= -3:
            return 0
        else:
            return x / 6. + 0.5


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
        if x > threshold:
            return x
        else:
            return 0
