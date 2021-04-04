import torch
import numpy as np

# ================================================================== #
#                             Tensors                                #
# ================================================================== #

device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(f'# my_tensor:\n{my_tensor}')
print(f'# my_tensor.dtype:\n{my_tensor.dtype}')
print(f'# my_tensor.device:\n{my_tensor.device}')
print(f'# my_tensor.shape:\n{my_tensor.shape}')
print(f'# my_tensor.grad:\n{my_tensor.grad}')

# Other ways to initialize a tensor
a = torch.empty(size=(3, 3))  # the values are uninitialized
print(f'# a = torch.empty(size=(3, 3)):\n{a}')
b = torch.zeros((3, 3))  # a zero tensor
print(f'# b = torch.zeros((3, 3)):\n{b}')
c = torch.rand((3, 3))  # from (0, 1)
print(f'# c = torch.rand((3, 3)):\n{c}')
d = torch.randn((3, 3))  # from normal distribution N(0, 1)
print(f'# d = torch.randn((3, 3)):\n{d}')
e = torch.ones((3, 3))  # a tensor filled with the scalar value 1
print(f'# e = torch.ones((3, 3)):\n{e}')
f = torch.eye(3)  # a 2-D tensor with ones on the diagonal and zeros elsewhere (Identity).
print(f'# f = torch.eye(3):\n{f}')
g = torch.arange(1, 11, 1)  # 11 (end) is non-inclusive
print(f'# g = torch.arange(1, 11, 1):\n{g}')
h = torch.linspace(0, 1, 11)
print(f'# h = torch.linspace(0, 1, 10):\n{h}')

# Numpy array to tensor and vise versa
np_arr = np.zeros((5, 5))
print(np_arr, type(np_arr))
tensor_from_arr = torch.from_numpy(np_arr)
print(tensor_from_arr, type(tensor_from_arr))
arr_back = tensor_from_arr.numpy()
print(arr_back, type(arr_back))

# Converting dtypes
tensor = torch.arange(5)
print(tensor, tensor.dtype)  # torch.int64
print(tensor.long(), tensor.long().dtype)  # torch.int64
print(tensor.short(), tensor.short().dtype)  # torch.int16
print(tensor.half(), tensor.half().dtype)  # torch.float16
print(tensor.bool(), tensor.bool().dtype)  # torch.bool
print(tensor.float(), tensor.float().dtype)  # torch.float32

