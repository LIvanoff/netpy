import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class CSNet_torch(torch.nn.Module):
    def __init__(self, n_input, n_hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(n_input, n_hidden)
        self.activ1 = torch.torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_input, 9)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(9, 8)
        self.activ3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(8, 7)
        self.activ4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(7, 8)
        self.activ5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(8, 9)
        self.activ6 = torch.nn.ReLU()
        self.fc7 = torch.nn.Linear(9, 10)
        self.activ7 = torch.nn.ReLU()
        self.fc8 = torch.nn.Linear(10, 2)
        self.activ8 = torch.nn.Softmax()

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


train = pd.read_excel('output1.xlsx', engine='openpyxl')
data = train.values
threshold = 20000

y_train = torch.Tensor(data[:threshold, 10:12])
x_train = torch.Tensor(data[:threshold, :10])
y_val = torch.Tensor(data[threshold:, 10:12])
x_val = torch.Tensor(data[threshold:, :10])

model = CSNet_torch(10, 10)
model.train()
optimizer = torch.optim.SGD(model.parameters(),
                             lr=1.0e-3)

for epoch in range(10000):
    # for i in range(0, len(x_train)):
    optimizer.zero_grad()
    pred = model.forward(x_train)
    loss = F.cross_entropy(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch: ' + str(epoch) + ' loss: ' + str(loss))

    with torch.no_grad():
        pred_val = model.forward(x_val)
        count = torch.count_nonzero(torch.eq(torch.argmax(pred_val, dim=1), torch.argmax(y_val, dim=1)))
        acc = count / pred_val.size(dim=0)
        acc = acc.detach().numpy() * 100
        if epoch % 100 == 0:
            print(f'acc: {acc}')
