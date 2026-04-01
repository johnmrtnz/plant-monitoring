#%%
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import numpy as np

#%%

# loading in plant dataset
data_dir = "./data/plantvillage dataset/color"

# define data transfomrations
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

full_dataset = ImageFolder(data_dir, transform=train_transform)
# %%

# split intro train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
trainset, valset = random_split(full_dataset, [train_size, val_size])

#apply different transform to val set
valset.dataset.transform = val_transform

train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=0)

print(f"Training samples: {len(trainset)}")
print(f"Validation samples: {len(valset)}")
print(f"Number of classes: {len(full_dataset.classes)}")
print(f"Classes: {full_dataset.classes[:5]}...")  # Show first 5


#%%
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# extracting features
def get_embeddings(images):
    inputs = processor(images=images, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # use CLS token
    embeddings = outputs.last_hidden_state[:, 0 ,:]
    return embeddings.numpy()

# get embeddings for all images
all_embeddings = []
all_labels = []

for img, label in valset:
    emb = get_embeddings(img.unsqueeze(0))
    all_embeddings.append(emb)
    all_labels.append(label)


#%%

# need to consolidate the embeddings into one vector
all_embeddings = np.vstack(all_embeddings)  # or np.concatenate(all_embeddings, axis=0)

# Cluster
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(all_embeddings)

# Visualize with t-SNE
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(12,8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:,1],
                      c=all_labels, cmap='tab10', alpha=0.6)

plt.colorbar(scatter)
plt.title('DINOv2 Embeddings (t-SNE)')
plt.show()
# %%

# Validating Results
