from network.layers import Linear
import torch


class Optimizer(object):
    def __init__(self, model: object, lr: float = 1e-4):
        self.lr = lr
        self.model = model

    def step(self):
        """
        function performs a single optimization step (parameter update)
        """

    def zero_grad(self):
        with torch.no_grad():
            for var in vars(self.model).items():
                if isinstance(var[1], Linear):
                    if var[1].weight.grad is not None:
                        var[1].weight.grad = var[1].weight.grad.zero_()
                        var[1].weight.bias = var[1].bias.grad.zero_()


class SGD(Optimizer):
    def __init__(self, model: object, lr: float = 1e-4):
        super().__init__(model, lr)

    def step(self):
        with torch.no_grad():
            for var in vars(self.model).items():
                if isinstance(var[1], Linear):
                    var[1].weight -= torch.mul(var[1].weight.grad, self.lr)
                    var[1].bias -= torch.mul(var[1].bias.grad, self.lr)


class StepScheduler(object):
    lr: float

    def __init__(self, optimizer: object):
        self.optimizer = optimizer
        for var in vars(self.optimizer).items():
            if isinstance(var[1], float):
                self.lr = var[1].lr


class StepLR(StepScheduler):
    step_size: int
    gamma: float

    def __init__(self, optimizer: object, step_size: int, gamma: float):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        pass
