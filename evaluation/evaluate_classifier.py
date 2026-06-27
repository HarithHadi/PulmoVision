import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from backend.dependencies import ModelContainer

# Load model
models = ModelContainer()
models.load_tb_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

y_true = []
y_scores = []
y_pred = []

dataset_root = Path("test_images") / "TB_Chest_Radiography_Database"


folders = [
    (1, dataset_root / "Tuberculosis"),
    (0, dataset_root / "Normal"),
]


for label, folder in folders:
    image_files = list(folder.glob("*.png"))

    print(folder)
    print("Found", len(image_files), "images")

    for img_path in tqdm(image_files, desc=f"Processing {folder.name}", unit="image"):
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        results = models.tb_classifier.run_pipeline(tensor)

        prob = float(results["probs"][1])
        y_scores.append(prob)
        y_true.append(label)
        y_pred.append(1 if prob >= 0.5 else 0)

auc = roc_auc_score(y_true, y_scores)
print(f"\nAUC-ROC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "TB"]))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — PulmoVision TB Classifier")
plt.legend()
plt.savefig("roc_curve.png")
print("Saved roc_curve.png")