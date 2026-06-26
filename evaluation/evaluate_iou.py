import sys
sys.path.append('../backend')

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from dependencies import ModelContainer
import json

models = ModelContainer()
models.load_tb_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def compute_iou(cam_norm, bbox, threshold=0.5):
    h, w = cam_norm.shape
    cam_binary = (cam_norm >= threshold).astype(np.uint8)
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    # Normalize bbox to cam size
    orig_h, orig_w = 518, 518
    x1 = int(x1 * orig_w)
    y1 = int(y1 * orig_h)
    x2 = int(x2 * orig_w)
    y2 = int(y2 * orig_h)
    gt_mask[y1:y2, x1:x2] = 1
    intersection = np.logical_and(cam_binary, gt_mask).sum()
    union = np.logical_or(cam_binary, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

# Load bounding box annotations
# Format: {"image_name.jpg": [x1, y1, x2, y2]} normalized 0-1
# You need to create this file manually or use existing annotations
with open("annotations.json") as f:
    annotations = json.load(f)

iou_scores = []

for img_name, bbox in annotations.items():
    img_path = f"test_images/tb/{img_name}"
    image = Image.open(img_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    results = models.tb_classifier.run_pipeline(tensor)
    cam_norm = results["cam_norm"]
    iou = compute_iou(cam_norm, bbox)
    iou_scores.append(iou)
    print(f"{img_name}: IoU = {iou:.4f}")

print(f"\nMean IoU: {sum(iou_scores)/len(iou_scores):.4f}")