import network.layers as nn
import network.loss_fn as F
import optimizer
import pandas as pd
import matplotlib.pyplot as plt
import torch.autograd
import numpy as np
import time


class CSNet(object):
    def __init__(self, n_input, n_hidden):
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, 9)
        self.activ2 = nn.ReLU()
        self.fc3 = nn.Linear(9, 8)
        self.activ3 = nn.ReLU()
        # self.fc4 = nn.Linear(8, 7)
        # self.activ4 = nn.ReLU()
        # self.fc9 = nn.Linear(7, 6)
        # self.activ9 = nn.ReLU()
        # self.fc10 = nn.Linear(6, 5)
        # self.activ10 = nn.ReLU()
        # self.fc11 = nn.Linear(5, 6)
        # self.activ11 = nn.ReLU()
        # self.fc12 = nn.Linear(6, 7)
        # self.activ12 = nn.ReLU()
        # self.fc5 = nn.Linear(7, 8)
        # self.activ5 = nn.ReLU()
        self.fc6 = nn.Linear(8, 9)
        self.activ6 = nn.ReLU()
        self.fc7 = nn.Linear(9, 10)
        self.activ7 = nn.ReLU()
        self.fc8 = nn.Linear(10, 2)
        self.activ8 = nn.Softmax()

        # self.activ1 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(n_input, 9)
        # self.activ2 = torch.nn.ReLU()
        # self.fc3 = torch.nn.Linear(9, 8)
        # self.activ3 = torch.nn.ReLU()
        # self.fc4 = torch.nn.Linear(8, 7)
        # self.activ4 = torch.nn.ReLU()
        # self.fc5 = torch.nn.Linear(7, 8)
        # self.activ5 = torch.nn.ReLU()
        # self.fc6 = torch.nn.Linear(8, 9)
        # self.activ6 = torch.nn.ReLU()
        # self.fc7 = torch.nn.Linear(9, 10)
        # self.activ7 = torch.nn.ReLU()
        # self.fc8 = torch.nn.Linear(10, 2)

        # self.activ1 = nn.Sigmoid()
        # self.fc2 = nn.Linear(n_hidden, 9)
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

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        out = self.fc3(out)
        out = self.activ3(out)
        # out = self.fc4(out)
        # out = self.activ4(out)
        # out = self.fc9(out)
        # out = self.activ9(out)
        # out = self.fc10(out)
        # out = self.activ10(out)
        # out = self.fc11(out)
        # out = self.activ11(out)
        # out = self.fc12(out)
        # out = self.activ12(out)
        # out = self.fc5(out)
        # out = self.activ5(out)
        out = self.fc6(out)
        out = self.activ6(out)
        out = self.fc7(out)
        out = self.activ7(out)
        out = self.fc8(out)
        out = self.activ8(out)
        return out


train = pd.read_excel('output.xlsx', engine='openpyxl')
data = train.values
threshold = 30000

y_train = torch.LongTensor(data[:threshold, 10:12])
x_train = torch.Tensor(data[:threshold, :10])
y_val = torch.LongTensor(data[threshold:, 10:12])
x_val = torch.Tensor(data[threshold:, :10])

csnet = CSNet(10, 10)
optim = optimizer.SGD(lr=0.0001, model=csnet)
scheduler = optimizer.StepLR(optim, step_size=10000, gamma=0.5)

loss_history_train = np.array([])
acc_history_val = np.array([])
x_loss = np.array([])

start_time = time.time()
for epoch in range(20000):
    optim.zero_grad()
    pred = csnet.forward(x_train)
    loss = F.CrossEntropy(y_train, pred)
    loss.backward()
    optim.step()
    scheduler.step()
    if epoch % 100 == 0:
        print(f'epoch: {epoch} loss: {loss}')

    loss_history_train = np.append(loss_history_train, loss.detach().numpy())
    x_loss = np.append(x_loss, epoch)

    with torch.no_grad():
        pred_val = csnet.forward(x_val)
        count = torch.count_nonzero(torch.eq(torch.argmax(pred_val, dim=1), torch.argmax(y_val, dim=1)))
        acc = count / pred_val.size(dim=0)
        acc = acc.detach().numpy() * 100
        if epoch % 100 == 0:
            print(f'acc: {acc}')
        acc_history_val = np.append(acc_history_val, acc)


print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(x_loss, loss_history_train, label='loss')
plt.plot(x_loss, acc_history_val, label='acc')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

with torch.no_grad():
    pred = csnet.forward(x_val)
    count = torch.count_nonzero(torch.eq(torch.argmax(pred, dim=1), torch.argmax(y_val, dim=1)))
    acc = count / pred.size(dim=0)
    print(f'Accuracy: {acc}')
    # x_pred = np.arange(pred.size(dim=0))
    #
    # plt.plot(x_pred, torch.argmax(y_train, dim=1), label='y', alpha=0.8)
    # plt.plot(x_pred, torch.argmax(pred, dim=1), 'r', label='pred', alpha=0.8)
    # plt.legend()
    # plt.grid(alpha=0.2)
    # plt.show()
