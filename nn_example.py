import torch
import torch.nn as nn  # Neural Networks
import torch.optim as optim  # Optimizers
import torch.nn.functional as F  # Functions
from torch.utils.data import DataLoader  # Load data, create mini-batches, etc
import torchvision.datasets as datasets  # https://pytorch.org/vision/stable/datasets.html
import torchvision.transforms as transforms

cnn = True


# Create a fully connected neural network
class Net(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dims, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create a CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super(CNN, self).__init__()
        # (none, 1, 28, 28) -> (none, 1, 28, 28): same convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # (none, 1, 28, 28) -> (none, 1, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # (none, 1, 14, 14) -> (none, 1, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Set devise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_dims = 784
output_dims = 10
lr = 0.001
batch_size = 8
epochs = 5

# Load Data
train_set = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
if cnn:
    model = CNN().to(device)
else:
    model = Net(input_dims, output_dims).to(device)

# Losses and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Calc Accuracy
def accuracy(loader, model):
    num_correct = 0
    num_sample = 0

    model.eval()

    with torch.no_grad():
        if loader.dataset.train:
            print('Train set')
        else:
            print('Test set')
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if not cnn:
                x = x.view(x.shape[0], -1)
            y_pred = model(x)
            _, prediction = y_pred.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
        acc = num_correct/num_sample
        print(f'Accuracy: {acc * 100:.2f}%')
        model.train()
        return acc


# Train the network
for _ in range(epochs):
    for index, (data, label) in enumerate(train_loader):
        data = data.to(device=device)
        label = label.to(device=device)
        if not cnn:
            data = data.view(batch_size, -1)
        # Forward
        pred = model(data)
        loss = criterion(pred, label)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index % 1000 == 0:
            print(index, '\t', loss)

    accuracy(test_loader, model)


accuracy(train_loader, model)
accuracy(test_loader, model)
