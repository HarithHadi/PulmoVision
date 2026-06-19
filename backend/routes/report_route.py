import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import UploadFile, File, APIRouter
from torchvision import transforms
from dependencies import models
from groq import Groq
from dotenv import load_dotenv
import re
import os

load_dotenv()

router = APIRouter()

IMAGE_SIZE = 518
COMPUTE_DTYPE = torch.float16

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


def clean_report(text: str) -> str:
    bad_phrases = [
        "prior", "previous", "interval", "since", "admission",
        "week earlier", "days ago", "CT scan", "compared to",
        "has been", "there has", "improvement since", "when there was",
        "was transferred", "transferred to", "ICU", "cardiology",
        "catheterization", "respiratory distress", "following cardiac",
        "he had", "she had", "patient was", "developed",
        "history of", "known case", "presenting with",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = [s for s in sentences if not any(p.lower() in s.lower() for p in bad_phrases)]
    return ' '.join(clean).strip()


def generate_report(visual_tokens: torch.Tensor, prediction: str, confidence: float) -> str:

    prompt = (
        f"TB classifier result: {prediction} ({confidence:.1f}% confidence).\n\n"
        f"Write a chest X-ray radiology report with exactly two sections: FINDINGS and IMPRESSION. "
        f"No headers, no bullet points, no patient information, no additional notes. "
        f"Plain text only. Start directly with FINDINGS:"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a radiologist writing concise radiology reports. "
                    "Output only FINDINGS and IMPRESSION sections in plain text. "
                    "No markdown, no bullet points, no bold text, no headers, no extra sections. "
                    "Never reference patient history, prior scans, or information not in this image."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=250,
        temperature=0.1,
    )

    report = response.choices[0].message.content.strip()

    # Strip markdown formatting
    report = re.sub(r'\*\*.*?\*\*', '', report)  # remove **bold**
    report = re.sub(r'\*.*?\*', '', report)        # remove *italic*
    report = re.sub(r'^[-•]\s+', '', report, flags=re.MULTILINE)  # remove bullet points
    report = re.sub(r'#+\s+', '', report)          # remove # headers
    report = re.sub(r'\n{3,}', '\n\n', report)     # collapse extra newlines

    # Keep only FINDINGS and IMPRESSION
    if "FINDINGS" in report:
        report = report[report.index("FINDINGS"):]
    if "Additional" in report:
        report = report[:report.index("Additional")].strip()
    if "Patient" in report:
        report = report[:report.index("Patient")].strip()

    report = clean_report(report)
    return report.strip()


@router.post("/report")
async def report(file: UploadFile = File(...)):
    classifier = models.tb_classifier
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    content = await file.read()
    image   = Image.open(io.BytesIO(content)).convert("RGB")
    tensor  = preprocess(image).unsqueeze(0).to(device)

    results       = classifier.run_pipeline(tensor)
    probs         = results["probs"]
    cam_norm      = results["cam_norm"]
    visual_tokens = results["visual_tokens"]

    prediction = "TB Positive" if int(probs.argmax()) == 1 else "Normal"
    confidence = float(probs[1]) * 100

    report_text = generate_report(visual_tokens, prediction, confidence)

    torch.cuda.empty_cache()

    return {
        "prediction":         "TB" if int(probs.argmax()) == 1 else "Normal",
        "tb_probability":     round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image":      build_overlay(image, cam_norm),
        "report":             report_text,
    }