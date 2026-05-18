"""
RAD-DINO TB Classifier + GradCAM Localization (Improved Visualization)
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import AutoModel

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 518          
BATCH_SIZE  = 8
EPOCHS      = 20
LR          = 1e-4
DATA_DIR    = Path("data/tb")
MODEL_SAVE  = Path("tb_classifier.pt")

# ── 1. Download ───────────────────────────────────────────────────────────────
def download_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MONTGOMERY_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
    SHENZHEN_URL   = "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"

    for name, url in [("Montgomery", MONTGOMERY_URL), ("Shenzhen", SHENZHEN_URL)]:
        zip_path = DATA_DIR / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {name}…")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR / name)

    normal_dir = DATA_DIR / "images" / "normal"
    tb_dir     = DATA_DIR / "images" / "tb"
    normal_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Montgomery & Shenzhen sorting
    for folder_name in ["Montgomery/MontgomerySet", "Shenzhen/ChinaSet_AllFiles"]:
        img_dir = DATA_DIR / folder_name / "CXR_png"
        for img_path in img_dir.glob("*.png"):
            label = img_path.stem[-1]
            dest = normal_dir if label == "0" else tb_dir
            shutil.copy(img_path, dest / img_path.name)
    print("Datasets ready.")

# ── 2. Dataset ───────────────────────────────────────────────────────────────
class TBDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = []
        self.transform = transform
        for label, folder in enumerate(["normal", "tb"]):
            for p in (root / folder).glob("*"):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((p, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# ── 3. Model ──────────────────────────────────────────────────────────────────
class RADDINOClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino", trust_remote_code=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        hidden_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        last_layer = self.backbone.encoder.layer[-1]
        def forward_hook(module, input, output): self._activations = output[0]
        def backward_hook(module, grad_input, grad_output): self._gradients = grad_output[0]
        last_layer.register_forward_hook(forward_hook)
        last_layer.register_full_backward_hook(backward_hook)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

    def get_gradcam(self, pixel_values, target_class=1):
        self.eval()
        for param in self.backbone.parameters(): param.requires_grad = True
        
        logits = self.forward(pixel_values)
        probs = logits.softmax(dim=-1)
        
        self.zero_grad()
        logits[0, target_class].backward()

        grads = self._gradients[0, 1:, :] 
        acts  = self._activations[0, 1:, :]
        
        weights = grads.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=-1))
        
        grid = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(grid, grid).detach().cpu().numpy()
        
        # Robust Normalization
        if (cam.max() - cam.min()) > 1e-7:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        for param in self.backbone.parameters(): param.requires_grad = False
        return cam, probs[0].detach().cpu().numpy(), logits.argmax(dim=-1).item()

# ── 4. Training ───────────────────────────────────────────────────────────────
def train():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = TBDataset(DATA_DIR / "images", transform=transform)
    n_val = int(len(dataset) * 0.2)
    train_set, val_set = random_split(dataset, [len(dataset)-n_val, n_val])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = RADDINOClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} complete.")
        torch.save(model.state_dict(), MODEL_SAVE)

# ── 5. Inference ──────────────────────────────────────────────────────────────
def infer(image_path: str, save_path: str):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = RADDINOClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Generate CAM for TB class (1)
    cam, probs, pred_idx = model.get_gradcam(tensor, target_class=1)

    # Prepare Heatmap
    orig_np = np.array(image)
    h, w = orig_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply a threshold to remove background "purple" noise
    cam_resized[cam_resized < 0.2] = 0 
    
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend: 60% original image, 40% heatmap
    blended = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    # UI Labels
    label = f"PRED: {'TB' if pred_idx==1 else 'Normal'} (TB Prob: {probs[1]*100:.1f}%)"
    cv2.putText(blended, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    final_output = np.hstack([orig_np, blended])
    cv2.imwrite(save_path, cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
    print(f"Inference saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--image", type=str)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    if args.download: download_datasets()
    elif args.train: train()
    elif args.infer: infer(args.image, args.output)