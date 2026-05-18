from fastapi import FastAPI, UploadFile
from transformers import pipeline, AutoImageProcessor, AutoModel
from PIL import Image
import io
import torch

app = FastAPI()

# 1. Initialize the model once when the server starts
model_id = "microsoft/rad-dino"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

@app.post("/diagnose")
async def diagnose_xray(file: UploadFile):
    # 2. Read the uploaded X-ray image
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content)).convert("RGB")

    # 3. Process the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # 4. Run Inference (Get Embeddings)
    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state contains the feature embeddings
    # Shape is [batch_size, sequence_length, hidden_size]
    embeddings = outputs.last_hidden_state

    #For a simple test, let's return the shape of the embeddings 
    # to confirm the model is working

    return{
        "model": "Rad-DINO",
        "embedding_shape": list(embeddings.shape),
        "messeage": "Feature extraction successful"
    }