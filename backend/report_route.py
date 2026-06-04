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

# Keep only the configuration constants
IMAGE_SIZE = 518
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def generate_report(visual_tokens: torch.Tensor) -> str:
    # Pull tokenizer and llama from the central 'models' container
    tokenizer = models.tokenizer
    llama = models.llama
    device = "cuda" if torch.cuda.is_available() else "cpu"

    instruction = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert radiologist. Analyze the chest X-ray and "
        "generate a structured radiology report.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Describe the findings and impression for this chest X-ray.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    inst_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
    text_embeds = llama.get_input_embeddings()(inst_ids).to(torch.bfloat16)
    
    visual_tokens = visual_tokens.to(torch.bfloat16)
    combined = torch.cat([visual_tokens, text_embeds], dim=1)
    attention_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = llama.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@router.post("/report")
async def report(file: UploadFile = File(...)):
    # Pull classifier from the container
    classifier = models.classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    results = classifier.run_pipeline(tensor)
    probs = results["probs"]
    cam_norm = results["cam_norm"]
    visual_tokens = results["visual_tokens"]

    report_text = generate_report(visual_tokens)
    torch.cuda.empty_cache()

    return {
        "prediction": "TB" if int(probs.argmax()) == 1 else "Normal",
        "tb_probability": round(float(probs[1]) * 100, 2),
        "normal_probability": round(float(probs[0]) * 100, 2),
        "overlay_image": build_overlay(image, cam_norm),
        "report": report_text,
    }