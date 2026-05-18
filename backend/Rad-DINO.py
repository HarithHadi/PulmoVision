from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel
from PIL import Image
from sklearn.decomposition import PCA
import torch
import numpy as np
import io
import base64
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load RAD-DINO once
model = AutoModel.from_pretrained(
    "microsoft/rad-dino",
    trust_remote_code=True
)
model = model.to(device)
model.eval()

# RAD-DINO expects 518x518 images (ViT-L/14 variant)
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

PATCH_SIZE = 14          # RAD-DINO patch size
GRID = 518 // PATCH_SIZE # = 37 → 37x37 patch grid


def extract_patch_tokens(image: Image.Image) -> np.ndarray:
    """Run image through RAD-DINO and return patch tokens [N, D]."""
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
    # last_hidden_state: [1, 1 + N_patches, D]
    # index 0 is [CLS], rest are spatial patch tokens
    patch_tokens = outputs.last_hidden_state[:, 1:, :]
    return patch_tokens.squeeze(0).cpu().numpy()  # [N, D]


def tokens_to_heatmap(patch_tokens: np.ndarray, grid: int) -> np.ndarray:
    """
    Use the first PCA component as an anomaly/foreground score.
    Returns a float32 heatmap in [0, 1] of shape (grid, grid).
    """
    pca = PCA(n_components=3)
    components = pca.fit_transform(patch_tokens)  # [N, 3]

    # First component captures most variance — correlates with
    # "unusual" regions when the model sees pathology.
    first_component = components[:, 0].reshape(grid, grid)

    # Normalise to [0, 1]
    lo, hi = first_component.min(), first_component.max()
    if hi - lo < 1e-6:
        return np.zeros((grid, grid), dtype=np.float32)

    heatmap = (first_component - lo) / (hi - lo)
    return heatmap.astype(np.float32)


def overlay_heatmap(original: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.55) -> str:
    """
    Blend the heatmap over the original image using a TURBO colormap.
    Returns the blended image as a base64 PNG string.
    """
    orig_w, orig_h = original.size

    # Upscale heatmap to original image size
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(
        heatmap_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
    )

    # Apply TURBO colormap (blue→green→yellow→red)
    colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Convert original to numpy
    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    overlay_np = colored_rgb.astype(np.float32)

    # Blend
    blended = ((1 - alpha) * orig_np + alpha * overlay_np).clip(0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended)

    # Encode to base64
    buf = io.BytesIO()
    blended_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/localizeRADDINO")
async def localize(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    patch_tokens = extract_patch_tokens(image)          # [N, D]
    heatmap = tokens_to_heatmap(patch_tokens, GRID)     # [37, 37]
    overlay_b64 = overlay_heatmap(image, heatmap)

    # Also return raw heatmap values for the frontend to draw its own viz
    heatmap_list = heatmap.tolist()

    return {
        "overlay_image": overlay_b64,   # base64 PNG
        "heatmap": heatmap_list,         # 2-D float array
        "grid_size": GRID,
    }