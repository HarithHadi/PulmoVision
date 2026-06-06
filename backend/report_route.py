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

IMAGE_SIZE = 518
COMPUTE_DTYPE = torch.float16   # T4: float16 only, no bfloat16

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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
    tokenizer    = models.tokenizer
    llama        = models.llama
    llama_device = next(llama.parameters()).device   # cuda on Colab

    instruction = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert radiologist. Analyze the chest X-ray and "
        "generate a structured radiology report.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Describe the findings and impression for this chest X-ray.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inst_ids    = tokenizer(instruction, return_tensors="pt").input_ids.to(llama_device)
    text_embeds = llama.get_input_embeddings()(inst_ids).to(COMPUTE_DTYPE)

    # Visual tokens: GPU + float16
    visual_tokens = visual_tokens.to(llama_device).to(COMPUTE_DTYPE)

    combined       = torch.cat([visual_tokens, text_embeds], dim=1)
    attention_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=llama_device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=COMPUTE_DTYPE):
            output_ids = llama.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
            )

    report = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "assistant" in report.lower():
        report = report.split("assistant")[-1].strip("\n |>")

    return report.strip()


@router.post("/report")
async def report(file: UploadFile = File(...)):
    classifier = models.classifier
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(device)

    results       = classifier.run_pipeline(tensor)
    probs         = results["probs"]
    cam_norm      = results["cam_norm"]
    visual_tokens = results["visual_tokens"]   # [1, 64, 3072] on GPU

    report_text = generate_report(visual_tokens)

    torch.cuda.empty_cache()

    return {
        "prediction":         "TB" if int(probs.argmax()) == 1 else "Normal",
        "tb_probability":     round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image":      build_overlay(image, cam_norm),
        "report":             report_text,
    }