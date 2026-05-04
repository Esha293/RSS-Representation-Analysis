import numpy as np

def generate_synthetic_data(num_classes=4, samples_per_class=50, dim=128):
    np.random.seed(42)

    data = []
    labels = []

    for i in range(num_classes):
        # Random mean for each class
        mean = np.random.uniform(-5, 5, dim)

        # Covariance matrix (identity scaled)
        cov = np.eye(dim) * np.random.uniform(0.5, 1.5)

        # Generate samples
        samples = np.random.multivariate_normal(mean, cov, samples_per_class)

        data.append(samples)
        labels.append(np.full(samples_per_class, i))

    data = np.vstack(data)
    labels = np.concatenate(labels)

    return data, labels


if __name__ == "__main__":
    X, y = generate_synthetic_data()

    print("Data shape:", X.shape)
    print("Labels shape:", y.shape)

    # Save for later use
    np.save("synthetic_features.npy", X)
    np.save("synthetic_labels.npy", y)

    print("Synthetic dataset saved successfully.")