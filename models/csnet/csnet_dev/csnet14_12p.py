import torch


class CSNet14_12p(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(12, 12)
        self.activ1 = torch.torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(12, 11)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(11, 10)
        self.activ3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(10, 9)
        self.activ4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(9, 8)
        self.activ5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(8, 7)
        self.activ6 = torch.nn.ReLU()
        self.fc7 = torch.nn.Linear(7, 6)
        self.activ7 = torch.nn.ReLU()
        self.fc8 = torch.nn.Linear(6, 7)
        self.activ8 = torch.nn.ReLU()
        self.fc9 = torch.nn.Linear(7, 8)
        self.activ9 = torch.nn.ReLU()
        self.fc10 = torch.nn.Linear(8, 9)
        self.activ10 = torch.nn.ReLU()
        self.fc11 = torch.nn.Linear(9, 10)
        self.activ11 = torch.nn.ReLU()
        self.fc12 = torch.nn.Linear(10, 11)
        self.activ12 = torch.nn.ReLU()
        self.fc13 = torch.nn.Linear(11, 12)
        self.activ13 = torch.nn.ReLU()
        self.fc14 = torch.nn.Linear(12, 2)
        self.activ14 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ1(out)
        indentity1 = out
        out = self.fc2(out)
        out = self.activ2(out)
        indentity2 = out
        out = self.fc3(out)
        out = self.activ3(out)
        indentity3 = out
        out = self.fc4(out)
        out = self.activ4(out)
        indentity4 = out
        out = self.fc5(out)
        out = self.activ5(out)
        indentity5 = out
        out = self.fc6(out)
        out = self.activ6(out)
        indentity6 = out
        out = self.fc7(out)
        out = self.activ7(out)
        out = self.fc8(out)
        out += indentity6
        out = self.activ8(out)
        out = self.fc9(out)
        out += indentity5
        out = self.activ9(out)
        out = self.fc10(out)
        out += indentity4
        out = self.activ10(out)
        out = self.fc11(out)
        out += indentity3
        out = self.activ11(out)
        out = self.fc12(out)
        out += indentity2
        out = self.activ12(out)
        out = self.fc13(out)
        out += indentity1
        out = self.activ13(out)
        out = self.fc14(out)
        out = self.activ14(out)
        return out
