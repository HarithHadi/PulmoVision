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
import torch.nn as nn
from PIL import Image
from pathlib import Path
from fastapi import UploadFile, File, APIRouter
from torchvision import transforms
from transformers import AutoModel

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE    = 518
PATCH_SIZE    = 14
GRID_SIZE     = IMAGE_SIZE // PATCH_SIZE   # 37
N_PATCHES     = GRID_SIZE * GRID_SIZE       # 1369
RAD_DINO_DIM  = 768
CLIP_DIM      = 512
LLAMA_DIM     = 4096
N_OUT_TOKENS  = 64
MODEL_SAVE    = Path("tb_classifier (5).pt")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── C-Abstractor ──────────────────────────────────────────────────────────────
class CAbstractor(nn.Module):
    def __init__(self, in_dim=CLIP_DIM, out_dim=LLAMA_DIM,
                 n_tokens=N_OUT_TOKENS, grid=GRID_SIZE):
        super().__init__()
        self.grid      = grid
        self.pool_size = int(n_tokens ** 0.5)  # 8

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B, N_PATCHES, CLIP_DIM]
        B, N, D = x.shape
        x = x.permute(0, 2, 1).reshape(B, D, self.grid, self.grid)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.proj(x)  # [B, N_OUT_TOKENS, LLAMA_DIM]


# ── RAD-DINO Classifier (EXACT same as tb_route.py) ──────────────────────────
class RADDINOClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/rad-dino", trust_remote_code=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._gradients   = None
        self._activations = None
        self._register_hooks()

        # Alignment + C-Abstractor on top
        self.align_proj  = nn.Linear(RAD_DINO_DIM, CLIP_DIM, bias=False)
        self.c_abstractor = CAbstractor()

    def _register_hooks(self):
        last_layer = self.backbone.encoder.layer[-1]

        def forward_hook(module, input, output):
            # NO .detach() — same as tb_route.py
            self._activations = output[0] if isinstance(output, tuple) else output

        def backward_hook(module, grad_input, grad_output):
            g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            self._gradients = g.detach()

        last_layer.register_forward_hook(forward_hook)
        last_layer.register_full_backward_hook(backward_hook)

    def forward(self, pixel_values):
        outputs   = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

    def run_pipeline(self, pixel_values):
        """
        EXACT same GradCAM as tb_route.py.
        Reuses the patch tokens for C-Abstractor — RAD-DINO runs ONCE.
        """
        self.eval()

        # Re-enable gradients — same as tb_route.py
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._gradients  = None
        self._activations = None

        pixel_values = pixel_values.to(DEVICE)
        logits = self.forward(pixel_values)
        probs  = logits.softmax(dim=-1)

        self.zero_grad()
        logits[0, 1].backward()  # TB class — same as tb_route.py

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire.")

        print(f"Grads norm: {self._gradients.norm():.6f}")

        # ── EXACT same GradCAM formula as tb_route.py ─────────────────────
        cls_grad   = self._gradients[0, 0, :]       # [D]
        patch_acts = self._activations[0, 1:, :]    # [N_patches, D]

        print(f"CLS grad norm: {cls_grad.norm():.6f}")

        cam = (patch_acts * cls_grad).sum(dim=-1)   # [N_patches]
        cam = torch.relu(cam)

        print(f"CAM raw: min={cam.min():.6f}, max={cam.max():.6f}")

        cam_grid = cam.reshape(GRID_SIZE, GRID_SIZE).detach().cpu().numpy()

        # ── EXACT same normalisation as tb_route.py ───────────────────────
        lo       = np.percentile(cam_grid, 60)
        hi       = np.percentile(cam_grid, 99)
        cam_norm = np.clip((cam_grid - lo) / (hi - lo + 1e-8), 0, 1)

        # ── C-Abstractor (new — reuses patch_acts, no second RAD-DINO) ────
        # Use GradCAM weights as attention mask on patch tokens
        cam_weights  = cam / (cam.mean() + 1e-8)            # [N_patches]
        cam_weights  = cam_weights.unsqueeze(-1)             # [N_patches, 1]
        weighted     = (patch_acts.detach() * cam_weights)   # [N_patches, D]
        weighted     = weighted.unsqueeze(0)                  # [1, N_patches, D]

        # Align RAD-DINO dim → CLIP dim
        clip_patches = self.align_proj(weighted)             # [1, N_patches, 512]

        # C-Abstractor → LLaMA tokens
        with torch.no_grad():
            visual_tokens = self.c_abstractor(clip_patches)  # [1, 64, 4096]

        # Re-freeze backbone — same as tb_route.py
        for param in self.backbone.parameters():
            param.requires_grad = False

        return {
            "probs":         probs[0].detach().cpu().numpy(),
            "cam_norm":      cam_norm,        # [37, 37] for overlay
            "visual_tokens": visual_tokens,   # [1, 64, 4096] for LLaMA
        }


# ── Load model at startup ─────────────────────────────────────────────────────
print("Loading PulmoVision model...")
model = RADDINOClassifier().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False),
    strict=False  # align_proj + c_abstractor not in old checkpoint, init randomly
)
model.eval()
print("PulmoVision model ready.")


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


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    results = model.run_pipeline(tensor)

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
    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    results       = model.run_pipeline(tensor)
    visual_tokens = results["visual_tokens"]  # [1, 64, 4096]

    t = visual_tokens[0]  # [64, 4096]

    return {
        "shape":        list(t.shape),
        "n_tokens":     t.shape[0],
        "token_dim":    t.shape[1],
        "mean":         round(float(t.mean()), 6),
        "std":          round(float(t.std()), 6),
        "min":          round(float(t.min()), 6),
        "max":          round(float(t.max()), 6),
        # First 3 tokens, first 16 dims each — readable preview
        "token_preview": t[:3, :16].tolist(),
        "note": "64 vectors of 4096 floats each. Prepend to LLaMA-3 input_embeds."
    }