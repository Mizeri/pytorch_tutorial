import torch

x = torch.arange(9)
print(x.view(-1, 3))
print(x.reshape(3, 3))
y = x.reshape(3, 3).t()
try:
    print(y.view(1, 9))
    print('view')
except RuntimeError:
    print(y.reshape(1, 9))
    print('reshape')

x1 = torch.randn(2, 5)
x2 = torch.randn(2, 5)
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)


