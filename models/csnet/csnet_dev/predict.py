import torch
from models.csnet.csnet_dev.csnet14_12p import CSNet14_12p


def predict(x: torch.Tensor):
    device = torch.device('cpu')
    new_model = CSNet14_12p()
    new_model.load_state_dict(torch.load('csnet14_12p.pth', map_location=device))
    new_model.eval()
    with torch.no_grad():
        return new_model(x)
        # count = torch.count_nonzero(torch.eq(torch.argmax(output, dim=1), torch.argmax(y_val, dim=1)))
        # acc = count / output.size(dim=0)
        # acc = acc.cpu() * 100
        # print(f'acc: {acc}')


# x = torch.Tensor([[52.9, 47.1, 8.0, 0.3125, 0.0625, 5., 6., 29.85,
#                    8.35, 3., 27.65, 20.05,  3.0000]])
x = torch.Tensor([[4., 23.0000, 11., 11., 5., 6., 29.85,
                   8.35, 3., 27.65, 20.05,  3.0000]])

# train = pd.read_excel('output4.xlsx', engine='openpyxl')
# data = train.values
# x = torch.Tensor([data[2, :12]])
print(predict(x))
