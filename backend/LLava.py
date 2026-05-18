from fastapi import FastAPI, UploadFile
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import io, torch

app = FastAPI()

model_id = "llava-med-v1.5-mistral-7b"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.post("/diagnoseLLava")
async def diagnose(file: UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    prompt = "USER: Analyze this chest X-ray. ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    output = model.generate(**inputs, max_new_tokens=100)
    result = processor.decode(output[0], skip_special_tokens=True)

    return {"result": result}


# This cannot be used because it uses heavy resources
