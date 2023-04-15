import neural_network as nn
import numpy as np


class Model(object):
    def __init__(self):
        self.fc1 = nn.Linear(3, 3)
        self.activ1 = nn.Sigmoid()
        self.fc2 = nn.Linear(3, 2)
        self.activ2 = nn.Sigmoid()
        self.fc3 = nn.Linear(2, 1)
        self.activ3 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        return out


x = np.array([0.1, 0.3, 1.])

model = Model()

model.forward(x)
