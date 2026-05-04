import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Use only first 5000 samples
subset_indices = list(range(5000))
dataset = Subset(dataset, subset_indices)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load pretrained ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()  # remove classification head
model = model.to(device)
model.eval()

features = []
labels = []

with torch.no_grad():
    for images, target in loader:
        images = images.to(device)
        output = model(images)
        features.append(output.cpu().numpy())
        labels.append(target.numpy())

features = np.vstack(features)
labels = np.hstack(labels)

np.save("simclr_cifar_features.npy", features)
np.save("simclr_cifar_labels.npy", labels)

print("SimCLR CIFAR features extracted")
print("Feature shape:", features.shape)