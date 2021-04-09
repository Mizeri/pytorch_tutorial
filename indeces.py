import torch

x = torch.arange(10)
print(x)
print(x.shape)

# 根据条件取元素
print(x[(x > 4) | (x < 1)])
print(x[(x > 3) & (x < 9)])

# reshape
print(x.view(-1, 5))
y = x.view(-1, 5)

# 切片操作
print(y[:, 0])
print(y[0, 1:2])
print(y[0, [0, 1]])

# 取余
print(y[y.remainder(3) == 0])

# 维数
print(y.ndimension())
z = torch.tensor([1, 1, 1, 4, 3, 6, 4, 3])
z = z.view(-1, 4)
print(z)

# 取不重复的元素
print(z.unique())

# Num of elements
print(z.numel())

