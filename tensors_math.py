import torch
import numpy as np

# ================================================================== #
#                             Tensors                                #
# ================================================================== #

# Initializing tensors
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[4, 5, 6], [7, 8, 9]])
print(f'# Initializing Tensor: \n{x}\n{y}', end='\n\n')

# Addition
print('# Addition:')
print(torch.add(x, y))
print(x + y, end='\n\n')  # The same

# Substraction
print('# Substraction:')
print(x - y, end='\n\n')

# Element wise multiplication
print('# Element wise multiplication')
print(x * y, end='\n\n')

# True division
print('# True division')
print(torch.true_divide(x, y))
print(torch.true_divide(x, 10), end='\n\n')

# Inplace
print('# Inplace Operation')
z = torch.zeros(2, 3)
print(z)
z.add_(x)
print(z, end='\n\n')

# Exponentiation
print('# Exponentiation')
print(x.pow(2))
print(x ** 2, end='\n\n')

# Compare
print('# Compare')
print(x > 5, end='\n\n')

# Matrix Multiplication
print('# MatMul')
print(torch.mm(x, y.T))
print(x.mm(y.T), end='\n\n')

# Matrix Exponentiation
print('# MatExp')
w = torch.randn(3, 3)
print(w)
print(w ** 3)
print(w.matrix_power(3))
print(torch.mm(torch.mm(w, w), w), end='\n\n')

# Dot
print('# Dot')
a = torch.arange(4)
b = torch.arange(4)
print(torch.dot(a, b), end='\n\n')

# Batch MatMul
print('# Batch MatMul')
m, n, p = 3, 4, 5
batch_size = 4
tensor1 = torch.randn((batch_size, m, n))
tensor2 = torch.randn((batch_size, n, p))
out_tensor = torch.bmm(tensor1, tensor2)
print(out_tensor, '\n', out_tensor.shape, end='\n\n')

# Broadcasting
print('# Broadcasting')
c = torch.randn((5, 5))
d = torch.randn((1, 5))
print(c)
print(d)
print(c - d, end='\n\n')

# Other Operations
print('# Sum')
print(torch.sum(x, dim=0))
print(torch.sum(x, dim=1), end='\n\n')

print('# Max')
print(torch.max(x))
values, indices = torch.max(x, dim=0)
print(values, indices)
values, indices = torch.max(x, dim=1)
print(values, indices, end='\n\n')

print('# Min')
print(torch.min(x))
values, indices = torch.min(x, dim=0)
print(values, indices)
values, indices = torch.min(x, dim=1)
print(values, indices, end='\n\n')

print('# Absolute value')
print(torch.abs(-x), end='\n\n')

print('# Argmax')
print(torch.argmax(x))
print(torch.argmax(x, dim=0))
print(torch.argmax(x, dim=1), end='\n\n')

print('# Mean')
print(torch.mean(x.float(), dim=0), end='\n\n')  # Float type required

print('# Equal')
print(torch.eq(x, y), end='\n\n')

print('# Sort')
print(torch.sort(x, dim=1, descending=True).values, end='\n\n')

print('# Clamp')
print(torch.clamp(x, min=2, max=5), end='\n\n')  # Set values that are less than min to min, greater than max to max

print('# Any')  # If one of elements is true
print(torch.any(x.bool()))

print('# All')  # If all of elements are true
print(torch.all(x.bool()))
print(x.bool().all())  # This kind of syntax can be applied to like max, min, mean...

