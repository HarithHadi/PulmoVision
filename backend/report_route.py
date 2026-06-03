"""
PulmoVision — Full Report Route
=================================
POST /report
    Input:  chest X-ray image
    Output: {
        prediction,
        tb_probability,
        normal_probability,
        overlay_image,      ← base64 PNG heatmap
        report,             ← generated radiology report text
    }
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
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

router = APIRouter()

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE    = 518
PATCH_SIZE    = 14
GRID_SIZE     = IMAGE_SIZE // PATCH_SIZE
N_PATCHES     = GRID_SIZE * GRID_SIZE
RAD_DINO_DIM  = 768
CLIP_DIM      = 512
LLAMA_DIM     = 4096
N_OUT_TOKENS  = 64
MODEL_SAVE    = Path("tb_classifier (5).pt")
STAGE1_PATH   = Path("pulmovision_stage1.pt")
LORA_PATH     = Path("lora_weights(curr)")
LLAMA_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"

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
        return self.proj(x)


# ── TB Classifier ─────────────────────────────────────────────────────────────
class RADDINOClassifier(nn.Module):
    def __init__(self, num_classes=2):
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

        self._gradients  = None
        self._activations = None

        pixel_values = pixel_values.to(DEVICE)
        logits = self.forward(pixel_values)
        probs  = logits.softmax(dim=-1)

        self.zero_grad()
        logits[0, 1].backward()

        cls_grad   = self._gradients[0, 0, :]
        patch_acts = self._activations[0, 1:, :]

        cam      = (patch_acts * cls_grad).sum(dim=-1)
        cam      = torch.relu(cam)
        cam_grid = cam.reshape(GRID_SIZE, GRID_SIZE).detach().cpu().numpy()

        lo       = np.percentile(cam_grid, 60)
        hi       = np.percentile(cam_grid, 99)
        cam_norm = np.clip((cam_grid - lo) / (hi - lo + 1e-8), 0, 1)

        cam_weights = cam / (cam.mean() + 1e-8)
        cam_weights = cam_weights.unsqueeze(-1)
        weighted    = (patch_acts.detach() * cam_weights).unsqueeze(0)

        clip_patches  = self.align_proj(weighted)
        with torch.no_grad():
            visual_tokens = self.c_abstractor(clip_patches)

        for param in self.backbone.parameters():
            param.requires_grad = False

        return {
            "probs":         probs[0].detach().cpu().numpy(),
            "cam_norm":      cam_norm,
            "visual_tokens": visual_tokens,
        }


# ── Load models at startup ────────────────────────────────────────────────────
print("Loading RAD-DINO classifier...")
classifier = RADDINOClassifier().to(DEVICE)
classifier.load_state_dict(
    torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False),
    strict=False
)
if STAGE1_PATH.exists():
    stage1 = torch.load(STAGE1_PATH, map_location=DEVICE, weights_only=False)
    classifier.align_proj.load_state_dict(stage1["align_proj"])
    classifier.c_abstractor.load_state_dict(stage1["c_abstractor"])
    print(f"Stage 1 loaded (val_loss={stage1['val_loss']:.4f})")
classifier.eval()
print("Classifier ready.")

print("Loading LLaMA-3 + LoRA...")

tokenizer = AutoTokenizer.from_pretrained(
    str(LORA_PATH) if LORA_PATH.exists() else LLAMA_MODEL
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add this config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 🚀 Swapped out 4-bit quantization for full native bfloat16
llama_base = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

if LORA_PATH.exists():
    llama = PeftModel.from_pretrained(llama_base, str(LORA_PATH))
    print("LoRA weights loaded.")
else:
    llama = llama_base
    print("WARNING: lora_weights/ not found — using base LLaMA-3.")

llama.eval()
print("LLaMA-3 ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_overlay(image: Image.Image, cam_norm: np.ndarray) -> str:
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


def generate_report(visual_tokens: torch.Tensor) -> str:
    instruction = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert radiologist. Analyze the chest X-ray and "
        "generate a structured radiology report.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Describe the findings and impression for this chest X-ray.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inst_ids    = tokenizer(instruction, return_tensors="pt").input_ids.to(DEVICE)
    text_embeds = llama.get_input_embeddings()(inst_ids).to(torch.bfloat16)
    
    # Match precision
    visual_tokens = visual_tokens.to(torch.bfloat16)
    combined      = torch.cat([visual_tokens, text_embeds], dim=1)

    # 🌟 FIX 1: Generate an attention mask of 1s matching the exact sequence length.
    # This tells LLaMA to pay full attention to both the image tokens and the text prompt.
    attention_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        output_ids = llama.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,          # 🌟 Pass the explicit mask
            pad_token_id=tokenizer.eos_token_id,    # 🌟 Fix the missing pad ID log warning
            max_new_tokens=300,
            
            # 🌟 FIX 2: Enable sampling to prevent rigid, robotic repeating loops
            do_sample=True,
            temperature=0.4,       # Low temperature keeps the language clinical and objective
            top_p=0.9,
            repetition_penalty=1.2 # Stronger penalty pushes the model away from repeating strings
        )
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/report")
async def report(file: UploadFile = File(...)):
    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(DEVICE)

    results       = classifier.run_pipeline(tensor)
    probs         = results["probs"]
    cam_norm      = results["cam_norm"]
    visual_tokens = results["visual_tokens"]

    label_names = ["Normal", "TB"]
    pred_class  = int(probs.argmax())

    print(f"Prediction: {label_names[pred_class]} ({probs[1]*100:.1f}% TB)")
    print("Generating report...")

    report_text = generate_report(visual_tokens)
    print("Report generated.")

    torch.cuda.empty_cache()

    return {
        "prediction":         label_names[pred_class],
        "tb_probability":     round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image":      build_overlay(image, cam_norm),
        "report":             report_text,
    }
