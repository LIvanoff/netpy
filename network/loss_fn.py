import torch


def MSE(y, y_hat):
    return torch.mean(torch.pow((y - y_hat), 2))


def RMSE(y, y_hat):
    return torch.sqrt(torch.mean(torch.pow((y - y_hat), 2)))


def MAE(y, y_hat):
    return torch.mean(torch.abs(y - y_hat))


def BCE(y, y_hat):
    y_hat = torch.clip(y_hat, 1e-10, 1 - 1e-10)
    return torch.mean(-(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))


def CrossEntropy(y, y_hat):
    y_hat = torch.clip(y_hat, 1e-10, 1 - 1e-10)
    return -torch.mean(y * torch.log(y_hat))
