import matplotlib.pyplot as plt

models = [
    "DenseNet121",
    "ResNet50",
    "TB-Net",
    "PulmoVision"
]

aurocs = [
    0.898,   
    0.949,   
    0.998,   # Replace
    0.891    # Your result
]

plt.figure(figsize=(10,6))

bars = plt.bar(models, aurocs)

plt.ylabel("AUROC")
plt.ylim(0.80, 1.00)
plt.title("Comparison of AUROC with Published TB Chest X-ray Classification Models")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.savefig("tb_auroc_comparison.png", dpi=300)
plt.show()