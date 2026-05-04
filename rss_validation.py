noise = np.random.normal(0, 0.01, features.shape)
noisy_features = features + noise

cov_noisy = np.cov(noisy_features, rowvar=False)
eigvals_noisy = np.linalg.eigvalsh(cov_noisy)

kappa_noisy = eigvals_noisy.max() / eigvals_noisy.min()

delta = abs(kappa_noisy - kappa) / kappa