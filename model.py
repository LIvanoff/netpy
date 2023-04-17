import network.layers as nn
import network.loss_fn as F
import optimizer
import torch.autograd


class Model(object):
    def __init__(self, input_size, hidden_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activ1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, 1)
        # self.activ2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        # out = self.fc2(out)
        # out = self.activ2(out)
        return out


x = torch.Tensor([[1.], [1.2], [1.6], [1.78], [2], [2.3], [2.4], [3], [3.3], [4.], [4.1], [4.12], [4.34], [5], [5.3], [5.6], [6]])
y = torch.Tensor([[0.8], [1], [0.9], [1.0], [1.2], [1.1], [1.6], [1.7], [2.0], [2.1], [2.15], [2.22], [2.45], [2.6], [2.12], [2.45], [2.3]])

model = Model(x.size(dim=0), 1)

optim = optimizer.SGD(lr=0.1, model=model)

for epoch in range(100):
    optim.zero_grad()
    pred = model.forward(x)
    loss = F.MSE(y, pred)
    print(f'loss: {loss}')
    loss.backward()
    optim.step()
