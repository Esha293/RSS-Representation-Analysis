import matplotlib.pyplot as plt

models = ["SimCLR", "DINO", "BYOL"]
rss = [1.27, 1.21, 1.02]

plt.figure(figsize=(6,4))

bars = plt.bar(models, rss)

# Value labels
for i, v in enumerate(rss):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title("Comparison of Representation Stability Score (RSS) Across SSL Methods", fontsize=12)

plt.ylim(0, 1.5)

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("rss_final.png", dpi=300)
plt.show()
