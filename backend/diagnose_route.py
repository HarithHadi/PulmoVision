"""
PulmoVision — Unified Diagnose Route
======================================
Keeps the EXACT same RAD-DINO + GradCAM logic from tb_route.py.
After GradCAM, patch tokens are passed through BiomedCLIP alignment
and C-Abstractor to produce LLaMA-3 visual tokens.

RAD-DINO runs ONCE — patch tokens are reused for both GradCAM and C-Abstractor.
"""
import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import UploadFile, File, APIRouter
from torchvision import transforms
from dependencies import models

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 518

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_overlay(image: Image.Image, cam_norm: np.ndarray) -> str:
    """EXACT same overlay as tb_route.py."""
    orig_w, orig_h = image.size
    cam_uint8   = (cam_norm * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    colored     = cv2.applyColorMap(cam_resized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    orig_np     = np.array(image.convert("RGB"), dtype=np.float32)
    blended     = (0.35 * orig_np + 0.65 * colored_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def serialize_tokens(tokens: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tokens.cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode()

@router.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    classifier = models.classifier   # ✅ use shared instance
    
    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    results = classifier.run_pipeline(tensor)   # ✅ was model.run_pipeline

    probs         = results["probs"]
    cam_norm      = results["cam_norm"]
    visual_tokens = results["visual_tokens"]

    label_names = ["Normal", "TB"]
    pred_class  = int(probs.argmax())

    print(f"Prediction: {label_names[pred_class]} ({probs[1]*100:.1f}% TB)")
    print(f"Visual tokens shape: {list(visual_tokens.shape)}")

    return {
        "prediction":         label_names[pred_class],
        "tb_probability":     round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image":      build_overlay(image, cam_norm),
        "visual_tokens":      serialize_tokens(visual_tokens),
        "token_shape":        list(visual_tokens.shape),
    }

@router.post("/debugTokens")
async def debug_tokens(file: UploadFile = File(...)):
    classifier = models.classifier   # ✅ use shared instance

    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    results       = classifier.run_pipeline(tensor)
    visual_tokens = results["visual_tokens"]   # [1, 64, 3072]
    t = visual_tokens[0]

    return {
        "shape":         list(t.shape),
        "n_tokens":      t.shape[0],
        "token_dim":     t.shape[1],             # should be 3072 now
        "mean":          round(float(t.mean()), 6),
        "std":           round(float(t.std()), 6),
        "min":           round(float(t.min()), 6),
        "max":           round(float(t.max()), 6),
        "token_preview": t[:3, :16].tolist(),
        "note": "64 vectors of 3072 floats each. Prepend to LLaMA-3.2-3B input_embeds."
    }