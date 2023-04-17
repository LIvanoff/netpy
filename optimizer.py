from torch.autograd import grad
import torch


class Optimizer(object):
    def __init__(self, lr: float = 0.001):
        self.lr = lr


class SGD(Optimizer):
    def __init__(self, lr, parameters: object):
        super().__init__(lr)

    def step(self):
        with torch.no_grad():
            # weight -= torch.mul(d_loss_dx_w, 0.01)
            # bias -= torch.mul(d_loss_dx_b, 0.01)
            pass

    def zero_grad(self):
        # .zero_()
        pass
