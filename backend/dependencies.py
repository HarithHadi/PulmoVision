# dependencies.py

import torch
from pathlib import Path
from .models_def import RADDINOClassifier, DEVICE

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "tb_classifier (5).pt"
)


class ModelContainer:
    def __init__(self):
        self.tb_classifier = None
        self.llama = None
        self.tokenizer = None

    def load_tb_model(self):
        if self.tb_classifier is None:
            print("Loading TB classifier once...")

            self.tb_classifier = RADDINOClassifier().to(DEVICE)

            state_dict = torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=False
            )

            self.tb_classifier.load_state_dict(state_dict, strict=False)
            self.tb_classifier.eval()

        return self.tb_classifier


models = ModelContainer()


def get_tb_model():
    return models.load_tb_model()