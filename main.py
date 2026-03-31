from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os
import uvicorn
import google.generativeai as genai

app = FastAPI()

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODEL ON STARTUP
# -------------------------
@app.on_event("startup")
def load_model():
    try:
        print("📥 Downloading / Loading model...")
        download_model()
        print("✅ Model Ready")
    except Exception as e:
        print("❌ MODEL LOAD ERROR:", str(e))

# -------------------------
# GEMINI SETUP
# -------------------------
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("⚠ GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):
    try:
        print("📸 Received file:", file.filename)

        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        print("📊 Running model prediction...")

        result = predict_nutrients(file_location, weight)

        print("✅ Prediction:", result)

        # delete temp file
        if os.path.exists(file_location):
            os.remove(file_location)

        return result

    except Exception as e:
        print("❌ PREDICT ERROR:", str(e))
        return {"error": str(e)}

# -------------------------
# RECOMMEND (GEMINI)
# -------------------------
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
    gender: str = Form("Unknown"),
    goal: str = Form("maintain"),
    disease: str = Form("")
):
    try:
        prompt = f"""
You are a professional nutrition AI.

User:
Age: {age}
Gender: {gender}
Goal: {goal}
Disease: {disease}

Nutrition:
Calories: {calories}
Protein: {protein}
Carbs: {carbohydrates}
Fats: {fats}
Fiber: {fiber}
Sugar: {sugars}
Sodium: {sodium}

Give 4-6 lines simple advice.
"""

        response = model.generate_content(prompt)

        return {
            "recommendations": [response.text]
        }

    except Exception as e:
        print("❌ GEMINI ERROR:", str(e))
        return {
            "error": str(e),
            "recommendations": ["AI failed"]
        }

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
