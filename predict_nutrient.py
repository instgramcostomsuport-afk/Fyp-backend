# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import json
# import os

# # -------------------------------
# # PATHS (Railway-safe)
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "models", "nutrifoodnet_final.h5")
# CLASS_LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.json")
# NUTRITION_CSV = os.path.join(BASE_DIR, "data", "nutrition.csv")

# IMAGE_SIZE = (299, 299)

# # -------------------------------
# # GLOBALS (lazy loaded)
# # -------------------------------
# model = None
# class_labels = None
# nutrition_df = None
# numeric_cols = ["weight","calories","protein","carbohydrates","fats","fiber","sugars","sodium"]

# # -------------------------------
# # LOADERS
# # -------------------------------
# def load_model_once():
#     global model
#     if model is None:
#         if not os.path.exists(MODEL_PATH):
#             raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
#         print("Loading ML model...")
#         model = tf.keras.models.load_model(
#     MODEL_PATH,
#     compile=False
# )
#         print("MODEL LOADED")

# def load_class_labels_once():
#     global class_labels
#     if class_labels is None:
#         with open(CLASS_LABELS_PATH, "r") as f:
#             class_labels = json.load(f)

# def load_nutrition_once():
#     global nutrition_df
#     if nutrition_df is None:
#         nutrition_df = pd.read_csv(NUTRITION_CSV)
#         nutrition_df[numeric_cols] = nutrition_df[numeric_cols].astype(float)

# # -------------------------------
# # PREDICTION FUNCTION
# # -------------------------------
# def predict_nutrients(img_path, target_weight=100):
#     load_model_once()
#     load_class_labels_once()
#     load_nutrition_once()

#     img = image.load_img(img_path, target_size=IMAGE_SIZE)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0) / 255.0

#     pred = model.predict(x)
#     pred_class_idx = int(np.argmax(pred))
#     confidence = float(pred[0][pred_class_idx])

#     food_name = class_labels[str(pred_class_idx)]

#     food_rows = nutrition_df[nutrition_df["label"] == food_name]
#     if food_rows.empty:
#         return {"error": "Nutrition info not found"}

#     food_rows = food_rows.reset_index(drop=True)
#     closest_row = food_rows.iloc[(food_rows['weight'] - target_weight).abs().argsort()[0]]

#     result = {
#         "label": food_name,
#         "confidence": round(confidence, 4),
#     }

#     for col in closest_row.index:
#         result[col] = float(closest_row[col]) if col in numeric_cols else closest_row[col]

#     return result




import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json
import os
import gdown

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "nutrifoodnet_final.h5")

CLASS_LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
NUTRITION_CSV = os.path.join(BASE_DIR, "data", "nutrition.csv")

IMAGE_SIZE = (299, 299)

# 🔴 YOUR GOOGLE DRIVE FILE ID HERE
MODEL_DRIVE_ID = "1ho8wwkADHIVGj1Iq5614A3h7uqbhkrTC"

# -------------------------------
# GLOBALS
# -------------------------------
model = None
class_labels = None
nutrition_df = None

numeric_cols = [
    "weight", "calories", "protein", "carbohydrates",
    "fats", "fiber", "sugars", "sodium"
]

# -------------------------------
# DOWNLOAD MODEL (ONLY ONCE)
# -------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")

        os.makedirs(MODEL_DIR, exist_ok=True)

        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

        gdown.download(url, MODEL_PATH, quiet=False)

        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists")

# -------------------------------
# LOAD MODEL (FIXED)
# -------------------------------
def load_model_once():
    global model

    if model is None:
        download_model()

        print("🚀 Loading ML model...")

        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False  # 🔥 FIX FOR batch_shape ERROR
            )

            print("✅ MODEL LOADED SUCCESSFULLY")

        except Exception as e:
            print(f"❌ MODEL LOAD FAILED: {e}")
            raise e

# -------------------------------
# LOAD LABELS
# -------------------------------
def load_class_labels_once():
    global class_labels

    if class_labels is None:
        with open(CLASS_LABELS_PATH, "r") as f:
            class_labels = json.load(f)

# -------------------------------
# LOAD NUTRITION DATA
# -------------------------------
def load_nutrition_once():
    global nutrition_df

    if nutrition_df is None:
        nutrition_df = pd.read_csv(NUTRITION_CSV)
        nutrition_df[numeric_cols] = nutrition_df[numeric_cols].astype(float)

# -------------------------------
# PREDICTION
# -------------------------------
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
        print(f"❌ Prediction error: {e}")
        return {"error": str(e)}
