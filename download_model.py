import os
import requests

MODEL_PATH = "models/nutrifoodnet_final.h5"
MODEL_URL = "https://drive.google.com/file/d/1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC/view?usp=sharing"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded")
    else:
        print("Model already exists")
