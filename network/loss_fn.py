import numpy as np


def MSE(y, y_hat):
    return np.mean(np.power((y - y_hat), 2))


def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.power((y - y_hat), 2)))


def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def BCE(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return -y * np.log(y_hat)