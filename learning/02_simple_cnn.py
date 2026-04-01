#%%

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader


#%% 

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        #convultional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 3 input channels, 32 output
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 32 input channels, 64 output
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1) # 64 input channels, 64 output

        # pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear( 64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10) # 10 classes

        # Dropout
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        # conv blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten
        x = x.view(-1, 64 * 4 * 4)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#%%

# initialize
model = SimpleCNN()
print(model)

# count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {total_params}")
# %%

# setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0010)

# Loading in CIFAR-10 dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root = './data', train=True ,
                                        download = True, transform = transform)

testset = torchvision.datasets.CIFAR10(root = './data', train=False,
                                       download = True, transform=transform)

# data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#%%
# training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total
    
# evaluation function

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

#%%

# running training for ten epochs

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, testloader, criterion, device)

    print(f"Epoch {epoch+1}/10")
    print(f"    Train Loss: {train_loss}, Train Acc: {train_acc}")
    print(f"    Val Loss: {val_loss}, Val Acc: {val_acc}")
# %%
