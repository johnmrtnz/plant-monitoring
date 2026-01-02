#%%

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader



#%%

# initialize
model = models.resnet18(pretrained=True)
print(model)

# count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {total_params}")

# feeeze early layers
for param in model.parameters():
    param.requires_grad = False

# replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # using 10 classes


# %%

# setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0010)

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

# running training for five epochs

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, testloader, criterion, device)

    print(f"Epoch {epoch+1}/10")
    print(f"    Train Loss: {train_loss}, Train Acc: {train_acc}")
    print(f"    Val Loss: {val_loss}, Val Acc: {val_acc}")
# %%
