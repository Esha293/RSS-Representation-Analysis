import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Starting BYOL feature extraction on CIFAR-10...")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])

    model = model.to(DEVICE)
    model.eval()

    print("ResNet50 backbone loaded!")

    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)

            feats = model(imgs)
            feats = feats.view(feats.size(0), -1)

            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)

    print("Feature extraction done!")
    print("Features shape:", features.shape)

    torch.save(features, "byol_cifar_features.pt")
    torch.save(labels, "byol_cifar_labels.pt")

    np.save("byol_cifar_features.npy", features.numpy())
    np.save("byol_cifar_labels.npy", labels.numpy())

    print("Saved features successfully!")

if __name__ == "__main__":
    main()