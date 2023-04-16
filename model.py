import network.layers as nn
import network.loss_fn as F
import numpy as np
import optimizer


class Model(object):
    def __init__(self):
        self.fc1 = nn.Linear(5, 5)
        self.activ1 = nn.Sigmoid()
        self.fc2 = nn.Linear(5, 4)
        self.activ2 = nn.Sigmoid()
        self.fc3 = nn.Linear(4, 5)
        self.activ3 = nn.Sigmoid()
        self.fc4 = nn.Linear(5, 1)
        self.activ4 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        out = self.fc4(out)
        out = self.activ4(out)
        return out


x = np.array([0.1, 0.3, 1., 0.4, 1.1])
y = np.array([1.])

model = Model()

# print(model.fc1.weight)
# print(model.fc2.weight)
# print(model.fc3.weight)
# print(model.fc4.weight)

optim = optimizer.SGD(lr=0.0001)

for epoch in range(1):
    pred = model.forward(x)
    loss = F.BCE(y, pred)
    print(pred)
    print(loss)