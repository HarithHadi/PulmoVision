import sys
sys.path.append('../backend')

import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')

load_dotenv('../backend/.env')

# Load OpenI reports
df = pd.read_csv("indiana_reports.csv")
df = df.dropna(subset=["findings", "impression"])
df["reference"] = "FINDINGS: " + df["findings"] + " IMPRESSION: " + df["impression"]

# Load your model + Groq
from dependencies import ModelContainer
models = ModelContainer()
models.load_tb_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

preprocess = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def generate_groq_report(prediction, confidence):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a radiologist. Output only FINDINGS and IMPRESSION in plain text. No markdown."
            },
            {
                "role": "user",
                "content": f"TB classifier result: {prediction} ({confidence:.1f}% confidence). Write FINDINGS and IMPRESSION only."
            }
        ],
        max_tokens=250,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Use first N images that have matching X-rays
IMAGE_DIR = "test_images/"  # put OpenI images here
N = 30  # number of pairs to evaluate

count = 0
for _, row in df.iterrows():
    if count >= N:
        break

    # Find matching image
    img_name = str(row.get("filename", "")).replace(".png", "").replace(".jpg", "")
    img_path = None
    for ext in [".png", ".jpg"]:
        p = Path(IMAGE_DIR) / (img_name + ext)
        if p.exists():
            img_path = p
            break

    if img_path is None:
        continue

    # Run classifier
    image = Image.open(img_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    results = models.tb_classifier.run_pipeline(tensor)
    prob = float(results["probs"][1])
    prediction = "TB Positive" if prob >= 0.5 else "Normal"
    confidence = prob * 100

    # Generate report via Groq
    generated = generate_groq_report(prediction, confidence)
    reference = row["reference"]

    # Score
    bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoother)
    rouge = scorer.score(reference, generated)

    bleu_scores.append(bleu)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rouge2_scores.append(rouge['rouge2'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

    print(f"[{count+1}/{N}] BLEU: {bleu:.3f} | ROUGE-L: {rouge['rougeL'].fmeasure:.3f}")
    count += 1

print("\n=== Report Quality Evaluation (OpenI) ===")
print(f"BLEU:    {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")