import numpy as np

def compute_metrics(file_path):
    print(f"\nProcessing: {file_path}")

    features = np.load(file_path)

    # Center features
    features = features - np.mean(features, axis=0)

    # Covariance
    cov = np.cov(features, rowvar=False)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    # Normalize
    p = eigenvalues / np.sum(eigenvalues)

    # Metrics
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)
    condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    rss = effective_rank / np.log(condition_number)

    print("Effective Rank:", effective_rank)
    print("Entropy:", entropy)
    print("Condition Number:", condition_number)
    print("RSS:", rss)

    return rss

# ---- CHANGE FILE PATH HERE ----
file_path = "Results/vicreg_cifar_features.npy"

compute_metrics(file_path)