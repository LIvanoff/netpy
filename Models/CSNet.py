import network.layers as nn
import network.loss_fn as F
import optimizer
import matplotlib.pyplot as plt
import torch.autograd
import numpy as np
import time


class CSNet(object):
    def __init__(self, n_input, n_hidden):
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(n_input, 9)
        self.activ2 = nn.ReLU()
        self.fc3 = nn.Linear(9, 8)
        self.activ3 = nn.ReLU()
        self.fc4 = nn.Linear(8, 7)
        self.activ4 = nn.ReLU()
        self.fc5 = nn.Linear(7, 8)
        self.activ5 = nn.ReLU()
        self.fc6 = nn.Linear(8, 9)
        self.activ6 = nn.ReLU()
        self.fc7 = nn.Linear(9, 10)
        self.activ7 = nn.ReLU()
        self.fc8 = nn.Linear(10, 2)

        # self.fc2 = nn.Linear(n_input, 9)
        # self.activ2 = nn.Sigmoid()
        # self.fc3 = nn.Linear(9, 8)
        # self.activ3 = nn.Sigmoid()
        # self.fc4 = nn.Linear(8, 7)
        # self.activ4 = nn.Sigmoid()
        # self.fc5 = nn.Linear(7, 8)
        # self.activ5 = nn.Sigmoid()
        # self.fc6 = nn.Linear(8, 9)
        # self.activ6 = nn.Sigmoid()
        # self.fc7 = nn.Linear(9, 10)
        # self.activ7 = nn.Sigmoid()
        # self.fc8 = nn.Linear(10, 2)

        self.activ8 = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        out = self.fc4(out)
        out = self.activ4(out)
        out = self.fc5(out)
        out = self.activ5(out)
        out = self.fc6(out)
        out = self.activ6(out)
        out = self.fc7(out)
        out = self.activ7(out)
        out = self.fc8(out)
        out = self.activ8(out)
        return out


x = torch.empty(size=(20000, 10)).normal_()
y = torch.eye(20000, 2)

csnet = CSNet(10, 10)
# optim = optimizer.SGD(lr=0.4, model=csnet)

loss_history = np.array([])
x_loss = np.array([])

start_time = time.time()
for epoch in range(100):
    # optim.zero_grad()
    pred = csnet.forward(x)
    loss = F.CrossEntropy(y, pred)
    # loss.backward()
    # optim.step()

    # if epoch % 100 == 0:
    print(f'epoch: {epoch} loss: {loss}')
    loss_history = np.append(loss_history, loss.detach().numpy())
    x_loss = np.append(x_loss, epoch)

print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(x_loss, loss_history)
plt.grid(alpha=0.2)
plt.show()