# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import os

# # Initialize FastAPI
# app = FastAPI()

# # Ensure the model exists (download if not)
# download_model()  # This should check if .h5 exists, otherwise download

# # Enable CORS so frontend can access backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Prediction endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...), weight: float = Form(...)):
#     # Save uploaded file temporarily
#     file_location = f"temp_{file.filename}"
#     with open(file_location, "wb") as f:
#         f.write(await file.read())
    
#     # Call your prediction function
#     result = predict_nutrients(file_location, weight)
    
#     # Clean up temporary file
#     if os.path.exists(file_location):
#         os.remove(file_location)
    
#     return result

# # Run server when executed directly (Render will use this)
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))  # Render sets $PORT
#     uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json

# =========================
# App Initialization
# =========================
app = FastAPI()

# Allow frontend / Expo / web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Model
# =========================
MODEL_PATH = "models/nutrifoodnet_final.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Load Labels
# =========================
with open("labels.json", "r") as f:
    labels = json.load(f)

labels = {int(k): v for k, v in labels.items()}

# =========================
# Nutrition Database (Simple)
# You can expand later
# =========================
nutrition_db = {
    "pizza": {"calories": 520, "protein": 18, "fat": 22},
    "hamburger": {"calories": 450, "protein": 20, "fat": 25},
    "apple_pie": {"calories": 300, "protein": 3, "fat": 14},
    "fried_rice": {"calories": 400, "protein": 8, "fat": 12},
    "ice_cream": {"calories": 280, "protein": 5, "fat": 15},
    "sushi": {"calories": 250, "protein": 12, "fat": 5},
    "salad": {"calories": 150, "protein": 5, "fat": 4},
}

# Default nutrition if not found
default_nutrition = {
    "calories": 250,
    "protein": 8,
    "fat": 10
}

# =========================
# Image Preprocessing
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# =========================
# Recommendation Logic
# =========================
def calculate_bmr(weight, height, age, gender):
    gender = gender.lower()
    if gender in ["male", "m"]:
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161


def adjust_calories(bmr, goal):
    goal = goal.lower()
    if goal == "weight_loss":
        return bmr - 500
    elif goal == "weight_gain":
        return bmr + 500
    elif goal == "muscle_gain":
        return bmr + 300
    else:
        return bmr


def generate_recommendation(food_label, nutrition, daily_calories, goal):
    calories = nutrition.get("calories", 0)
    protein = nutrition.get("protein", 0)

    if goal == "weight_loss":
        if calories > 400:
            advice = f"{food_label} is high in calories. Eat smaller portions or choose grilled options."
        else:
            advice = f"{food_label} is suitable for weight loss in moderation."

    elif goal == "weight_gain":
        advice = f"{food_label} helps increase calorie intake. Add protein-rich sides."

    elif goal == "muscle_gain":
        if protein < 15:
            advice = f"{food_label} is low in protein. Add eggs, chicken, or yogurt."
        else:
            advice = f"{food_label} is good for muscle building."

    else:
        advice = f"{food_label} can be part of a balanced diet."

    return {
        "daily_calories_target": round(daily_calories, 2),
        "advice": advice
    }


# =========================
# API Endpoints
# =========================

@app.get("/")
def home():
    return {"message": "Food Nutrition API is running"}


# -------- 1. Predict Food + Nutrition --------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    food_label = labels[class_index]

    # Get nutrition
    nutrition = nutrition_db.get(food_label, default_nutrition)

    return {
        "food_label": food_label,
        "confidence": confidence,
        "nutrition": nutrition
    }


# -------- 2. Recommendation after Nutrition --------
@app.post("/recommend")
async def recommend(data: dict = Body(...)):
    food_label = data.get("food_label")
    nutrition = data.get("nutrition")
    weight = float(data.get("weight"))
    height = float(data.get("height"))
    age = int(data.get("age"))
    gender = data.get("gender")
    goal = data.get("goal")

    # Step 1: BMR
    bmr = calculate_bmr(weight, height, age, gender)

    # Step 2: Adjust calories
    daily_calories = adjust_calories(bmr, goal)

    # Step 3: Advice
    result = generate_recommendation(
        food_label,
        nutrition,
        daily_calories,
        goal
    )

    return result
