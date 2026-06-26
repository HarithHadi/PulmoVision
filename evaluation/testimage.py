import os
from pathlib import Path
print("Current Working Directory:", os.getcwd())
print("Does the path exist?", Path("test_images/images").exists())