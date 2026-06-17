# models_def.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import numpy as np
import os

IMAGE_SIZE    = 518
PATCH_SIZE    = 14
GRID_SIZE     = IMAGE_SIZE // PATCH_SIZE
N_PATCHES     = GRID_SIZE * GRID_SIZE
RAD_DINO_DIM  = 768
CLIP_DIM      = 512
LLAMA_DIM     = 3072
N_OUT_TOKENS  = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAbstractor(nn.Module):
    def __init__(self, in_dim=CLIP_DIM, out_dim=LLAMA_DIM,
                 n_tokens=N_OUT_TOKENS, grid=GRID_SIZE):
        super().__init__()
        self.grid      = grid
        self.pool_size = int(n_tokens ** 0.5)

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
        B, N, D = x.shape
        x = x.permute(0, 2, 1).reshape(B, D, self.grid, self.grid)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.proj(x)  # [B, N_OUT_TOKENS, 3072]


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

        self.align_proj   = nn.Linear(RAD_DINO_DIM, CLIP_DIM, bias=False)
        self.c_abstractor = CAbstractor()

    def _register_hooks(self):
        last_layer = self.backbone.encoder.layer[-1]

        def forward_hook(module, input, output):
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
        self.eval()

        for param in self.backbone.parameters():
            param.requires_grad = True

        self._gradients   = None
        self._activations = None

        pixel_values = pixel_values.to(DEVICE)
        logits = self.forward(pixel_values)
        probs  = logits.softmax(dim=-1)

        self.zero_grad()
        logits[0, 1].backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not fire.")

        print(f"Grads norm: {self._gradients.norm():.6f}")

        cls_grad   = self._gradients[0, 0, :]
        patch_acts = self._activations[0, 1:, :]

        print(f"CLS grad norm: {cls_grad.norm():.6f}")

        cam = (patch_acts * cls_grad).sum(dim=-1)
        cam = torch.relu(cam)

        print(f"CAM raw: min={cam.min():.6f}, max={cam.max():.6f}")

        cam_grid = cam.reshape(GRID_SIZE, GRID_SIZE).detach().cpu().numpy()

        lo       = np.percentile(cam_grid, 60)
        hi       = np.percentile(cam_grid, 99)
        cam_norm = np.clip((cam_grid - lo) / (hi - lo + 1e-8), 0, 1)

        cam_weights  = cam / (cam.mean() + 1e-8)
        cam_weights  = cam_weights.unsqueeze(-1)
        weighted     = (patch_acts.detach() * cam_weights)
        weighted     = weighted.unsqueeze(0)

        clip_patches_report = self.align_proj(patch_acts.detach().unsqueeze(0))

        with torch.no_grad():
            visual_tokens = self.c_abstractor(clip_patches_report)  # [1, 64, 3072] ✅

        for param in self.backbone.parameters():
            param.requires_grad = False

        return {
            "probs":         probs[0].detach().cpu().numpy(),
            "cam_norm":      cam_norm,
            "visual_tokens": visual_tokens,   # [1, 64, 3072]
        }


def load_all_models(device):
    # ── Classifier ────────────────────────────────────────────────────────────
    classifier = RADDINOClassifier().to(device)
    classifier.load_state_dict(
        torch.load("tb_classifier (5).pt", map_location=device, weights_only=False),
        strict=False
    )

    # ── Stage 1 weights ───────────────────────────────────────────────────────────
    stage1_path = Path("stage1_checkpoint.pt")
    if stage1_path.exists():
        stage1 = torch.load(stage1_path, map_location=device, weights_only=False)

        # Remap "proj.weight" → "weight" for align_proj
        align_state = {}
        for k, v in stage1["align_proj"].items():
            new_key = k.replace("proj.", "", 1)   # "proj.weight" → "weight"
            align_state[new_key] = v

        classifier.align_proj.load_state_dict(align_state)
        classifier.c_abstractor.load_state_dict(stage1["c_abstractor"])
        print(f"✓ Stage 1 weights loaded (val_loss={stage1['val_loss']:.4f})")
    else:
        print("⚠ Stage 1 checkpoint not found — reports will be gibberish!")

    classifier.eval()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # ── LLaMA 4-bit ───────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    llama_base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    llama = PeftModel.from_pretrained(llama_base, "lora_weights(curr)")
    llama.eval()

    return classifier, llama, tokenizer