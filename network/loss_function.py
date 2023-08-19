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


class BCELoss(object):
    def __init__(self):
        pass

    def __call__(self, pred, y, *args, **kwargs):
        pred = torch.clip(pred, 1e-10, 1 - 1e-10)
        return pred - y * y + torch.log(1 + torch.exp(-pred))


class CrossEntropyLoss(object):
    def __init__(self):
        pass

    def __call__(self, pred, y, *args, **kwargs):
        pred = torch.clip(pred, 1e-10, 1 - 1e-10)
        return -torch.mean(y * torch.log(pred))
