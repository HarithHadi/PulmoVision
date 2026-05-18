"""
FastAPI router — RAD-DINO TB inference + GradCAM overlay

FIXES vs original:
    - forward hook does NOT call .detach() on activations
    - GradCAM formula corrected: channel_weights = grads.mean(dim=0)
    - activation/gradient buffers are cleared before each call
"""

from fastapi import UploadFile, File, APIRouter
from transformers import AutoModel
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
import io
import base64
from pathlib import Path

router = APIRouter()

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 518
MODEL_SAVE = Path("tb_classifier (5).pt")


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

        self._gradients  = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        last_layer = self.backbone.encoder.layer[-1]

        def forward_hook(module, input, output):
            # ✓ NO .detach() — must keep computation graph intact
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

    def get_gradcam(self, pixel_values, target_class: int = 1):
        self.eval()

        # Re-enable gradients for backbone so hooks fire
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._gradients  = None
        self._activations = None

        pixel_values = pixel_values.to(DEVICE)
        logits = self.forward(pixel_values)
        probs  = logits.softmax(dim=-1)

        self.zero_grad()
        logits[0, target_class].backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire.")

        print(f"Grads norm: {self._gradients.norm():.6f}")

        # CLS token gradient [D] — this is where the signal actually is
        cls_grad = self._gradients[0, 0, :]        # [D]
        # Patch activations [N_patches, D]
        patch_acts = self._activations[0, 1:, :]   # [N_patches, D]

        print(f"CLS grad norm: {cls_grad.norm():.6f}")

        # Weight patch activations by CLS gradient
        cam = (patch_acts * cls_grad).sum(dim=-1)  # [N_patches]
        cam = torch.relu(cam)

        print(f"CAM raw: min={cam.min():.6f}, max={cam.max():.6f}")

        print(f"CAM range: [{cam.min():.6f}, {cam.max():.6f}]") 

        print(f"CAM range: [{cam.min():.6f}, {cam.max():.6f}]")

        grid = int(cam.shape[0] ** 0.5)
        cam  = cam.reshape(grid, grid).detach().cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            print("WARNING: CAM is flat — gradients may be vanishing.")

        # Re-freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        return cam, probs[0].detach().cpu().numpy()


# ── Load once at startup ──────────────────────────────────────────────────────
print("Loading TB classifier...")
tb_model = RADDINOClassifier().to(DEVICE)
tb_model.load_state_dict(
    torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False)
)
tb_model.eval()
print("TB classifier ready.")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@router.post("/diagnoseTB")
async def diagnose_tb(file: UploadFile = File(...)):
    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    cam, probs = tb_model.get_gradcam(tensor, target_class=1)

    label_names = ["Normal", "TB"]
    pred_class  = int(probs.argmax())

    orig_w, orig_h = image.size
    cam_uint8   = (cam * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    colored     = cv2.applyColorMap(cam_resized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image.convert("RGB"), dtype=np.float32)
    blended = (0.45 * orig_np + 0.55 * colored_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prediction":         label_names[pred_class],
        "tb_probability":     round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image":      overlay_b64,
    }
