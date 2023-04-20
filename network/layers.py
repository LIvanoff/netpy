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


class BatchNormalize(object):
    """
    layer for normalizing the input of a neural network
    """

    def __call__(self, x):
        return x


class Sigmoid(object):
    def __call__(self, x: torch.Tensor):
        return 1. / (1 + torch.exp(-x))


class ReLU(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1,)
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] < 0:
        #             x[i][j] = 0


class ELU(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor, alpha: float = 1.):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = alpha * (torch.exp(x[i]) - 1)
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] < 0:
        #             x[i][j] = alpha * (torch.exp(x[i][j]) - 1)


class LeakyReLU(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor, alpha: float = 0.01):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = alpha * x[i]
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] < 0:
        #             x[i][j] = alpha * x[i][j]


class sign(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] > 0:
                x[i] = 1
            elif x[i] == 0:
                x[i] = 0
            else:
                x[i] = -1
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] > 0:
        #             x[i][j] = 1
        #         elif x[i][j] == 0:
        #             x[i][j] = 0
        #         else:
        #             x[i][j] = -1


class Heaviside(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor, value: float):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] > 0:
                x[i] = 1
            elif x[i] == 0:
                x[i] = 0
            else:
                x[i] = -1
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] > 0:
        #             x[i][j] = 1
        #         elif x[i][j] == 0:
        #             x[i][j] = value
        #         else:
        #             x[i][j] = -1


class Hardshrink(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor, lambda_: float = 0.5):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] <= lambda_ or x[i] >= -lambda_:
                x[i] = 0
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] <= lambda_ or x[i][j] >= -lambda_:
        #             x[i][j] = 0


class Hardsigmoid(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            if x[i] >= 3:
                x[i] = 1
            elif x[i] <= -3:
                x[i] = 0
            else:
                x[i] = x[i] / 6. + 0.5
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         if x[i][j] >= 3:
        #             x[i][j] = 1
        #         elif x[i][j] <= -3:
        #             x[i][j] = 0
        #         else:
        #             x[i][j] = x[i][j] / 6. + 0.5


class SoftPlus(object):
    def __call__(self, x: torch.Tensor):
        return torch.log(1 + torch.exp(x))


class Softmax(object):
    def __call__(self, x: torch.Tensor):
        return torch.exp(x) / torch.sum(torch.exp(x), axis=1, keepdims=True)


class LogSigmoid(object):
    def __call__(self, x: torch.Tensor):
        return torch.log(1. / (1. + torch.exp(-x)))


class Tanh(object):
    def __call__(self, x: torch.Tensor):
        return torch.tanh(x)


class Threshold(object):
    ''' Нужно отрефакторить '''
    def __call__(self, x: torch.Tensor, threshold: float, value: float = 0.):
        x_dim0 = x.size(dim=0)
        x_dim1 = x.size(dim=1)
        x = x.view(-1, )
        for i in range(len(x)):
            x[i] = 1. if x[i] > threshold else value
        x = x.view(x_dim0, x_dim1)
        return x
        # for i in range(x.size(dim=0)):
        #     for j in range(x.size(dim=1)):
        #         x[i][j] = 1. if x[i][j] > threshold else value
        # return x
