import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VICReg-style transform (stronger augmentations)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
])

dataset = CIFAR10(root="data/cifar", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Backbone
model = models.resnet50(weights="DEFAULT")

# Remove classifier
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

features = []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        out = out.view(out.size(0), -1)
        features.append(out.cpu().numpy())

features = np.concatenate(features, axis=0)

np.save("Results/vicreg_cifar_features.npy", features)

print("VICReg features done:", features.shape)