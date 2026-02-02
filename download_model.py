import os
import gdown

MODEL_DIR = "models"
MODEL_NAME = "nutrifoodnet_final.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 🔴 Replace with YOUR Google Drive file ID
MODEL_URL = "https://drive.google.com/file/d/1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC/view?usp=sharing"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found. Downloading from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists")
