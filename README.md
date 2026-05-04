# Representation Stability Score (RSS)

This repository contains the implementation for the paper:

**"Geometric and Spectral Analysis of Self-Supervised Representations"**

---

## 📌 Overview

We propose a metric called **Representation Stability Score (RSS)** to analyze:

- Effective Rank (intrinsic dimensionality)
- Spectral Entropy (feature distribution)
- Condition Number (numerical stability)

---

## ⚙️ Setup

```bash
pip install numpy scipy torch torchvision matplotlib
## Code Structure

- `rss.py` → Original implementation used in paper
- `compute_rss.py` → Simplified reference version
- `generate_synthetic.py` → Synthetic dataset
- `extract_*` → Feature extraction scripts
