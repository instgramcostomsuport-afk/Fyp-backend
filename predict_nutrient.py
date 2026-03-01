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
#         model = tf.keras.models.load_model(MODEL_PATH)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "nutrifoodnet_final.h5")
CLASS_LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.json")
NUTRITION_CSV = os.path.join(BASE_DIR, "data", "nutrition.csv")

IMAGE_SIZE = (299, 299)

model = None
class_labels = None
nutrition_df = None

numeric_cols = [
    "weight", "calories", "protein",
    "carbohydrates", "fats", "fiber",
    "sugars", "sodium"
]


def load_model_once():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found")
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded")


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
    load_model_once()
    load_class_labels_once()
    load_nutrition_once()

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])

    food_name = class_labels[str(idx)]

    rows = nutrition_df[nutrition_df["label"] == food_name]
    if rows.empty:
        return {"error": "Nutrition not found"}

    rows = rows.reset_index(drop=True)
    closest = rows.iloc[(rows["weight"] - target_weight).abs().argsort()[0]]

    result = {
        "label": food_name,
        "confidence": round(confidence, 4)
    }

    for col in closest.index:
        if col in numeric_cols:
            result[col] = float(closest[col])
        else:
            result[col] = closest[col]

    return result
