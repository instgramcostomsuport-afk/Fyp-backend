import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json
import sys

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/nutrifoodnet_final.h5"
CLASS_LABELS_PATH = "models/class_labels.json"
NUTRITION_CSV = "data/nutrition.csv"
IMAGE_SIZE = (299, 299)  # Model input size

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("MODEL LOADED")

# -------------------------------
# LOAD CLASS LABELS
# -------------------------------
with open(CLASS_LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# -------------------------------
# LOAD NUTRITION DATA
# -------------------------------
nutrition_df = pd.read_csv(NUTRITION_CSV)
# Convert numeric columns to float
numeric_cols = ["weight","calories","protein","carbohydrates","fats","fiber","sugars","sodium"]
nutrition_df[numeric_cols] = nutrition_df[numeric_cols].astype(float)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_nutrients(img_path, target_weight=100):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # (1, 299, 299, 3)
    x = x / 255.0  # normalize

    # Predict
    pred = model.predict(x)
    pred_class_idx = np.argmax(pred)
    confidence = pred[0][pred_class_idx]
    food_name = class_labels[str(pred_class_idx)]

    print(f"\nPredicted food: {food_name}, Confidence: {confidence:.2f}")

    # Find closest weight in nutrition CSV
    food_rows = nutrition_df[nutrition_df["label"] == food_name]

    if food_rows.empty:
        print("Nutrition info not found for this food.")
        return

    # Reset index to avoid KeyError
    food_rows = food_rows.reset_index(drop=True)

    closest_row = food_rows.iloc[(food_rows['weight'] - target_weight).abs().argsort()[0]]

    # Display nutrition
    print("\nNutrition info (closest weight):")
    for col in closest_row.index:
        if col in numeric_cols:
            print(f"{col}: {closest_row[col]:.2f}")
        else:
            print(f"{col}: {closest_row[col]}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_nutrient.py <image_path> [target_weight_in_grams]")
        sys.exit(1)

    img_path = sys.argv[1]
    target_weight = float(sys.argv[2]) if len(sys.argv) >= 3 else 100  # default 100g
    predict_nutrients(img_path, target_weight)
