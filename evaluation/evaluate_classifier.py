import sys
sys.path.append('../backend')

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from dependencies import ModelContainer
from tqdm import tqdm

# Initialize the PulmoVision model container
models = ModelContainer()
models.load_tb_model() # Loads the frozen Rad-DINO backbone
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

y_true = []
y_scores = []
y_pred = []

dataset_root = Path("TB_Chest_Radiography_Database")

class_mapping = {
    "Normal": 0,
    "Tuberculosis": 1
}

for class_name, label in class_mapping.items():

    class_dir = dataset_root / class_name

    print(f"Processing {class_name}...")

    for img_path in tqdm(class_dir.glob("*.png"), desc=class_name):

        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        results = models.tb_classifier.run_pipeline(tensor)
        prob = float(results["probs"][1])

        y_scores.append(prob)
        y_true.append(label)
        y_pred.append(1 if prob >= 0.5 else 0)

# Failsafe in case the directory path is incorrect
if not y_true:
    print("No images processed. Please verify your dataset_dir path.")
    sys.exit()

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