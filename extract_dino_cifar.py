import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
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
    download=False,
    transform=transform
)

subset_indices = list(range(5000))
dataset = Subset(dataset, subset_indices)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load DINO model from torch hub
model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
model.fc = torch.nn.Identity()
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

np.save("dino_cifar_features.npy", features)
np.save("dino_cifar_labels.npy", labels)

print("DINO CIFAR features extracted")
print("Feature shape:", features.shape)