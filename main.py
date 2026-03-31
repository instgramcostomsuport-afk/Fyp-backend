from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os
import google.generativeai as genai

app = FastAPI()

# -----------------------------
# Load Model
# -----------------------------
download_model()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 🔥 GEMINI CONFIG
# -----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")


# -----------------------------
# AI FUNCTION
# -----------------------------
def generate_ai_recommendation(nutrition, goal, disease, age, gender):

    prompt = f"""
You are a professional nutrition expert.

User Info:
Age: {age}
Gender: {gender}
Goal: {goal}
Disease: {disease}

Nutrition:
Calories: {nutrition['calories']}
Protein: {nutrition['protein']}
Carbohydrates: {nutrition['carbohydrates']}
Fats: {nutrition['fats']}
Fiber: {nutrition['fiber']}
Sugars: {nutrition['sugars']}
Sodium: {nutrition['sodium']}

Give response in this format:
1. Health Verdict
2. Reason
3. 3 practical suggestions
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"


# -----------------------------
# Predict Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    result = predict_nutrients(file_location, weight)

    if os.path.exists(file_location):
        os.remove(file_location)

    return result


# -----------------------------
# Recommendation Endpoint
# -----------------------------
@app.post("/recommend")
async def recommend(
    calories: float = Form(...),
    protein: float = Form(...),
    carbohydrates: float = Form(...),
    fats: float = Form(...),
    fiber: float = Form(...),
    sugars: float = Form(...),
    sodium: float = Form(...),

    age: int = Form(0),
    gender: str = Form("unknown"),
    goal: str = Form("maintain"),
    disease: str = Form("")
):

    nutrition = {
        "calories": calories,
        "protein": protein,
        "carbohydrates": carbohydrates,
        "fats": fats,
        "fiber": fiber,
        "sugars": sugars,
        "sodium": sodium
    }

    ai_response = generate_ai_recommendation(
        nutrition,
        goal,
        disease,
        age,
        gender
    )

    return {
        "recommendations": [ai_response],
        "status": "success"
    }


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Nutrition AI Running 🚀"}
