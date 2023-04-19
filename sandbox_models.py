import network.layers as nn
import network.loss_fn as F
import optimizer
import torch.autograd
import matplotlib.pyplot as plt
import numpy as np
import time

"""
class Regression(object):
    def __init__(self, n_input, n_hidden):
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.activ1 = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        return out

    @staticmethod
    def plot():
        plt.clf()
        plt.scatter(x, y, marker='o', alpha=0.8)
        plt.plot(x, pred.detach().numpy(), 'r')
        plt.grid(alpha=0.2)
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.0001)


x = torch.Tensor([[1.], [1.2], [1.6], [1.78], [2], [2.3], [2.4], [3], [3.3], [4.], [4.1], [4.12], [4.34], [5], [5.3], [5.6], [6]])
y = torch.Tensor([[0.8], [1], [0.9], [1.0], [1.2], [1.1], [1.6], [1.7], [2.0], [2.1], [2.15], [2.22], [2.45], [2.6], [2.12], [2.45], [2.3]])

# x = torch.Tensor([0., 1., 2.3, 0.6])
# y = torch.Tensor([1])

model = Regression(x.size(dim=1), 1)

optim = optimizer.SGD(lr=0.01, model=model)

plt.ion()
for epoch in range(100):
    optim.zero_grad()
    pred = model.forward(x)
    loss = F.MSE(y, pred)
    loss.backward()
    optim.step()
    if epoch % 10 == 0:
        print(f'loss: {loss}')

    model.plot()


plt.ioff()
plt.show()

with torch.no_grad():
    pred = model.forward(x)

plt.scatter(x, y, marker='o', alpha=0.8)
plt.plot(x, pred.detach().numpy(), 'r')
plt.grid(alpha=0.2)
plt.show()"""


class LogisticRegression(object):
    def __init__(self, n_input, n_hidden):
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.activ1 = nn.Sigmoid()
        self.fc2 = nn.Linear(3, 1)
        self.activ2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        return out

    @staticmethod
    def plot(pred, pcolor):
        plt.clf()
        plt.tricontourf(xx.ravel(), yy.ravel(), pcolor, cmap=plt.get_cmap('Blues'), alpha=1)
        plt.scatter(x[:, 0], x[:, 1], alpha=0.8, c=pred.detach().numpy())
        plt.grid(alpha=0.2)
        plt.draw()
        plt.gcf().canvas.flush_events()


x = torch.Tensor(
    [[1., 1.1],
     [1., 3.2],
     [1.4, 2.1],
     [2., 2.9],
     [2.1, 4.2],
     [2.5, 1.1],
     [2.3, 2.1],
     [2.9, 3.4],
     [3.1, 2.6],
     [3.4, 4.1],
     [3.8, 1.5],
     [3.9, 3.1],

     [5.2, 5.],
     [5.6, 6.],
     [5.7, 7.],
     [5.5, 9.],
     [5.3, 8.2],
     [6.1, 6.2],
     [6.3, 7.2],
     [6.1, 8.5],
     [7., 7.1],
     [7.3, 6.8],
     [6.6, 7.3],
     [8.1, 6.2],
     [8.4, 6.7],
     [7.9, 8.2]])
y = torch.Tensor(
    [[0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1],
     [1]])

eps = 0.1
xx, yy = np.meshgrid(np.linspace(0.7 - eps, 8.75 + eps, 5),
                     np.linspace(0.71 - eps, 9.37 + eps, 5))
z = torch.Tensor(np.array([xx.ravel(), yy.ravel()]))
z = z.T

classificator = LogisticRegression(2, 3)

optim = optimizer.SGD(lr=0.1, model=classificator)

plt.ion()
for epoch in range(2000):
    optim.zero_grad()
    pred = classificator.forward(x)
    loss = F.BCE(y, pred)
    loss.backward()
    optim.step()
    if epoch % 10 == 0:
        print(f'loss: {loss}')

    with torch.no_grad():
        pcolor = classificator.forward(z)
        pcolor = torch.reshape(pcolor, (-1,))

    classificator.plot(pred, pcolor)

plt.ioff()
plt.show()

with torch.no_grad():
    pcolor = classificator.forward(z)
    pcolor = torch.reshape(pcolor, (-1,))
    pred = classificator.forward(x)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[1].tricontourf(xx.ravel(), yy.ravel(), pcolor, cmap=plt.get_cmap('Blues'), alpha=1)
axs[0].scatter(x[:, 0], x[:, 1], alpha=0.8, c=y)
axs[1].scatter(x[:, 0], x[:, 1], alpha=0.8, c=pred)
axs[0].grid(alpha=0.2)
axs[1].grid(alpha=0.2)
plt.show()