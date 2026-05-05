# Representation Stability Score (RSS)

Implementation for the paper:
**"Geometric and Spectral Analysis of Self-Supervised Representations"**

---

## 📌 Overview

This repository provides code to compute the **Representation Stability Score (RSS)**, a metric that captures:

* Intrinsic dimensionality (Effective Rank)
* Spectral diversity (Entropy)
* Numerical stability (Condition Number)

---

## ⚙️ Setup

Install dependencies:

```
pip install numpy scipy torch torchvision matplotlib
```

---

## ▶️ Reproducibility

### Synthetic experiment

```
python generate_synthetic.py
python compute_rss.py
```

---

## 📂 Code Structure

* `rss_original.py` → Original implementation used in paper
* `compute_rss.py` → Simplified reference version
* `rss_validation.py` → Validation experiments
* `rss_compare.py` → Comparison across methods
* `extract_simclr_cifar.py` → SimCLR feature extraction
* `extract_dino_cifar.py` → DINO feature extraction
* `generate_synthetic.py` → Synthetic dataset generator

---

## 📊 Notes

* Large files (`.npy`, `.pt`) are excluded
* All results can be reproduced using the provided scripts

---

## 🔗 Paper & Code

Code: https://github.com/Esha293/RSS-Representation-Analysis

