import os
import gdown

MODEL_PATH = "models/nutrifoodnet_final.h5"
MODEL_DIR = "models"
GDRIVE_FILE_ID = "1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size < 1_000_000:
            print("⚠️ Model corrupted, re-downloading...")
            os.remove(MODEL_PATH)
        else:
            print(f"✅ Model exists — {size / 1_000_000:.1f} MB")
            return

    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("❌ Model download failed")

    size = os.path.getsize(MODEL_PATH)
    if size < 1_000_000:
        raise RuntimeError(f"❌ File too small ({size} bytes) — Google returned HTML, not model")

    print(f"✅ Model downloaded — {size / 1_000_000:.1f} MB")
