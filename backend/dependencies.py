# dependencies.py
class ModelContainer:
    def __init__(self):
        self.classifier = None
        self.llama = None
        self.tokenizer = None

models = ModelContainer()