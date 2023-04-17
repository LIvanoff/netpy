import network.layers as nn
import network.loss_fn as F
import optimizer
import torch
import torch.autograd


class Model(object):
    def __init__(self, input_size, hidden_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activ2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        out = self.fc2(out)
        out = self.activ2(out)
        return out


x = torch.Tensor([0.1, 0.3, 0.12, 0.5, 0.7])
y = torch.Tensor([1.])

model = Model(x.size(dim=0), 2)
# print(model.fc2.weight)
# print(model.fc1.weight)
# print(model.fc2.weight)
# print(model.fc3.weight)
# print(model.fc4.weight)

# print(type(model))

optim = optimizer.SGD(lr=0.0001, model=model)

for epoch in range(10):
    pred = model.forward(x)
    loss = F.MSE(y, pred)
    loss.backward()
    optim.step()
    # print(model.fc2.weight.grad)
    # print("pred " + str(pred))
    # print("y " + str(y))
    # print("loss " + str(loss))
