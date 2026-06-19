# dependencies.py
import torch
from models_def import RADDINOClassifier, DEVICE


class ModelContainer:
    def __init__(self):
        self.tb_classifier = None
        self.llama = None
        self.tokenizer = None

    def load_tb_model(self):
        if self.tb_classifier is None:
            print("Loading TB classifier once...")
            self.tb_classifier = RADDINOClassifier().to(DEVICE)
            # Add weights_only=False if you are loading local .pt files
            state_dict = torch.load("models/tb_classifier (5).pt", map_location=DEVICE, weights_only=False)
            self.tb_classifier.load_state_dict(state_dict, strict=False)
            self.tb_classifier.eval()
        return self.tb_classifier

# Create one instance for the whole app
models = ModelContainer()

# Helper for FastAPI Dependency Injection
def get_tb_model():
    return models.load_tb_model()