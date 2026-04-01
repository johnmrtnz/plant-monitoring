#%%

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#%%

# Loading in CIFAR-10 dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root = './data', train=True ,
                                        download = True, transform = transform)

testset = torchvision.datasets.CIFAR10(root = './data', train=False,
                                       download = True, transform=transform)

#%%

# Vizualizing data

fig, axes = plt.subplots(2, 5, figsize=(12,5))

for i, ax in enumerate(axes.flat):
    img, label = trainset[i]
    ax.imshow(img.permute(1,2,0))
    ax.set_title(trainset.classes[label])
    ax.axis('off')

plt.show()
# %%
