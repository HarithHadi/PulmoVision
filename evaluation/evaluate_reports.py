import sys
sys.path.append('../backend')

import os
import json
from dotenv import load_dotenv
load_dotenv('../backend/.env')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')

# Pairs of (generated, reference) reports
# Fill these with your actual generated reports vs radiologist references
report_pairs = [
    {
        "generated": "FINDINGS: The chest X-ray demonstrates increased opacity in the right upper lobe. IMPRESSION: Findings are consistent with active pulmonary tuberculosis.",
        "reference": "FINDINGS: Right upper lobe consolidation with cavitation noted. IMPRESSION: Active TB likely."
    },
    # Add more pairs here — ideally 20-30
]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for pair in report_pairs:
    gen = pair["generated"]
    ref = pair["reference"]

    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smoother)
    rouge = scorer.score(ref, gen)

    bleu_scores.append(bleu)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rouge2_scores.append(rouge['rouge2'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

print("\n=== Report Quality Evaluation ===")
print(f"BLEU:    {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")