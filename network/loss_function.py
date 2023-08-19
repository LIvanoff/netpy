import torch


class MSELoss(object):
    def __init__(self):
        pass

    def __call__(self, pred, y, *args, **kwargs):
        return torch.mean(torch.pow((pred - y), 2))


class RMSELoss(object):
    def __init__(self):
        self.epsilon = 1e-6

    def __call__(self, pred, y, *args, **kwargs):
        return torch.sqrt(torch.mean(torch.pow((pred - y), 2))) + self.epsilon


class MAELoss(object):
    def __init__(self):
        pass

    def __call__(self, pred, y, *args, **kwargs):
        return torch.mean(torch.abs(pred - y))


def BCE(y, y_hat):
    y_hat = torch.clip(y_hat, 1e-10, 1 - 1e-10)
    return torch.mean(-(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))


def CrossEntropy(y, y_hat):
    y_hat = torch.clip(y_hat, 1e-10, 1 - 1e-10)
    return -torch.mean(y * torch.log(y_hat))
