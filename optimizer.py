import autograd as ad
import autograd.variable as av


class Optimizer(object):
    def __init__(self, lr: float = 0.001):
        self.lr = lr


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)
