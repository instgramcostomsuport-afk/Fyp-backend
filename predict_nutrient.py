import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json
import os
import gdown
import h5py
from tensorflow.keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "nutrifoodnet_final.h5")

CLASS_LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
NUTRITION_CSV = os.path.join(BASE_DIR, "data", "nutrition.csv")

IMAGE_SIZE = (299, 299)

MODEL_DRIVE_ID = "1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC"

model = None
class_labels = None
nutrition_df = None

numeric_cols = [
    "weight", "calories", "protein", "carbohydrates",
    "fats", "fiber", "sugars", "sodium"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model download failed")
        size = os.path.getsize(MODEL_PATH)
        if size < 1_000_000:
            raise RuntimeError(f"Downloaded file too small ({size} bytes)")
        print(f"Model downloaded — {size / 1_000_000:.1f} MB")
    else:
        size = os.path.getsize(MODEL_PATH)
        if size < 1_000_000:
            print("Model file corrupted, re-downloading...")
            os.remove(MODEL_PATH)
            download_model()
        else:
            print(f"Model already exists — {size / 1_000_000:.1f} MB")

def load_model_once():
    global model
    if model is None:
        download_model()
        print("Loading ML model...")
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
            print("MODEL LOADED SUCCESSFULLY")
        except Exception as e:
            print(f"Load failed: {e}")
            raise e

def load_class_labels_once():
    global class_labels
    if class_labels is None:
        with open(CLASS_LABELS_PATH, "r") as f:
            class_labels = json.load(f)

def load_nutrition_once():
    global nutrition_df
    if nutrition_df is None:
        nutrition_df = pd.read_csv(NUTRITION_CSV)
        nutrition_df[numeric_cols] = nutrition_df[numeric_cols].astype(float)

def predict_nutrients(img_path, target_weight=100):
    try:
        load_model_once()
        load_class_labels_once()
        load_nutrition_once()

        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        pred = model.predict(x)

        pred_class_idx = int(np.argmax(pred))
        confidence = float(pred[0][pred_class_idx])

        food_name = class_labels[str(pred_class_idx)]

        food_rows = nutrition_df[nutrition_df["label"] == food_name]

        if food_rows.empty:
            return {"error": "Nutrition info not found"}

        food_rows = food_rows.reset_index(drop=True)

        closest_row = food_rows.iloc[
            (food_rows['weight'] - target_weight).abs().argsort()[0]
        ]

        result = {
            "label": food_name,
            "confidence": round(confidence, 4),
        }

        for col in closest_row.index:
            result[col] = float(closest_row[col]) if col in numeric_cols else closest_row[col]

        return result

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}
