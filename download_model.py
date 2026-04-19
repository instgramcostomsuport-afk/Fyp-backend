# import os
# import gdown

# MODEL_PATH = "models/nutrifoodnet_final.h5"
# MODEL_DIR = "models"
# GDRIVE_FILE_ID = "1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC"

# def download_model():
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     if os.path.exists(MODEL_PATH):
#         size = os.path.getsize(MODEL_PATH)
#         if size < 1_000_000:
#             print("⚠️ Model corrupted, re-downloading...")
#             os.remove(MODEL_PATH)
#         else:
#             print(f"✅ Model exists — {size / 1_000_000:.1f} MB")
#             return

#     print("📥 Downloading model from Google Drive...")
#     url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
#     gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

#     if not os.path.exists(MODEL_PATH):
#         raise RuntimeError("❌ Model download failed")

#     size = os.path.getsize(MODEL_PATH)
#     if size < 1_000_000:
#         raise RuntimeError(f"❌ File too small ({size} bytes) — Google returned HTML, not model")

#     print(f"✅ Model downloaded — {size / 1_000_000:.1f} MB")



import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "models/nutrifoodnet_final.h5"
MODEL_DIR = "models"
GDRIVE_FILE_ID = "1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download if needed
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size < 50_000_000:  # 50MB se kam ho to suspicious
            print("⚠️ Model corrupted, re-downloading...")
            os.remove(MODEL_PATH)
        else:
            print(f"✅ Model exists — {size / 1_000_000:.1f} MB")
    else:
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("❌ Model download failed - file not found")
        
        size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model downloaded — {size / 1_000_000:.1f} MB")
    
    # =============== MODEL LOAD ===============
    print("🔄 Loading the model...")
    try:
        # Important changes yahan:
        model = load_model(
            MODEL_PATH, 
            compile=False,           # ← Yeh bohot important hai
            # custom_objects={       # Agar custom layers hain to yahan add karo
            #     'YourCustomLayer': YourCustomLayer,
            #     ...
            # }
        )
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        return model
        
    except Exception as e:
        print(f"❌ Model loading FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
