import numpy as np

def compute_metrics(features, name="Model"):
    # Covariance
    cov = np.cov(features, rowvar=False)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)

    # Effective Rank
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p))
    reff = np.exp(entropy)

    # Condition Number
    kappa = eigvals.max() / eigvals.min()

    # RSS
    rss = reff / np.log(kappa)

    # Noise Stability
    noise = np.random.normal(0, 0.01, features.shape)
    noisy_features = features + noise

    cov_noisy = np.cov(noisy_features, rowvar=False)
    eigvals_noisy = np.linalg.eigvalsh(cov_noisy)
    eigvals_noisy = np.clip(eigvals_noisy, 1e-10, None)

    kappa_noisy = eigvals_noisy.max() / eigvals_noisy.min()

    delta_kappa = abs(kappa_noisy - kappa) / kappa

    # Print nicely
    print(f"\n===== {name} =====")
    print("Effective Rank:", reff)
    print("Condition Number:", kappa)
    print("RSS:", rss)
    print("Stability Change (Δκ):", delta_kappa)

    return reff, kappa, rss, delta_kappa


# ==============================
# 🔹 LOAD YOUR FEATURES HERE

# Example:
# simclr_features = np.load("simclr_features.npy")
# dino_features = np.load("dino_features.npy")

# TEMP (REMOVE THESE AFTER LOADING REAL DATA)
simclr_features = np.load("simclr_cifar_features.npy")
dino_features = np.load("dino_cifar_features.npy")

# ==============================

# Run both
simclr_results = compute_metrics(simclr_features, "SimCLR")
dino_results = compute_metrics(dino_features, "DINO")