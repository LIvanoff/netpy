from network.layers import Linear
import torch


class Optimizer(object):
    def __init__(self, model: object, lr: float = 1e-4):
        self.lr = torch.Tensor([lr])
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
                    var[1].weight -= torch.mul(var[1].weight.grad, self.lr[0])
                    var[1].bias -= torch.mul(var[1].bias.grad, self.lr[0])


class LRScheduler(object):
    lr: torch.Tensor
    count: int

    def __init__(self, optimizer: object):
        self.optimizer = optimizer
        self.count = 0
        for var in vars(self.optimizer).items():
            if isinstance(var[0], str):
                self.lr = var[1]
                return


class StepLR(LRScheduler):
    step_size: int
    gamma: float

    def __init__(self, optimizer: object, step_size: int, gamma: float):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.count += 1

        if self.count % self.step_size == 0:
            self.lr[0] *= self.gamma

