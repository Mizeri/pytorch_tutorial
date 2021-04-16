import torch
import torch.nn as nn  # Neural Networks
import torch.optim as optim  # Optimizers
import torch.nn.functional as F  # Functions
from torch.utils.data import DataLoader  # Load data, create mini-batches, etc
import torchvision.datasets as datasets  # https://pytorch.org/vision/stable/datasets.html
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
