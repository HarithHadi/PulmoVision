from fastapi import FastAPI, UploadFile, File
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import torch
import io

app = FastAPI()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once (IMPORTANT)
model, preprocess = create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
tokenizer = get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

model = model.to(device)
model.eval()

# Labels (you can tweak these later)
labels = [
    "a normal chest radiograph",
    "a chest radiograph showing tuberculosis",
    "a chest radiograph showing pneumonia"
]

text_tokens = tokenizer(labels).to(device)


@app.post("/diagnoseBiomed")
async def diagnose(file: UploadFile = File(...)):
    content = await file.read()

    # Load image
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, text_tokens)
        probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    # Convert to readable output
    results = {
        labels[i]: float(probs[0][i])
        for i in range(len(labels))
    }

    return {
        "predictions": results,
        "top_prediction": labels[probs.argmax().item()]
    }