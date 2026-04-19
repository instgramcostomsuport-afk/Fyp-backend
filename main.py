


# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import google.generativeai as genai
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Gemini setup
# # Gemini setup (FIXED)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("❌ GEMINI_API_KEY not set in environment variables")

# genai.configure(api_key=GEMINI_API_KEY)

# gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# @app.on_event("startup")
# def startup_event():
#     try:
#         print("Downloading model check...")
#         download_model()
#         print("Model ready")
#     except Exception as e:
#         print(f"Startup error: {e}")

# @app.get("/")
# def home():
#     return {"message": "Backend running"}

# def generate_ai_recommendation(nutrition, goal, disease):
#     try:
#         prompt = f"""
# You are a professional nutritionist AI.

# User Goal: {goal}
# Disease/Condition: {disease}

# Nutrition Values Detected:
# - Calories: {nutrition.get('calories')} kcal
# - Protein: {nutrition.get('protein')} g
# - Carbohydrates: {nutrition.get('carbohydrates')} g
# - Fats: {nutrition.get('fats')} g
# - Fiber: {nutrition.get('fiber')} g
# - Sugars: {nutrition.get('sugars')} g
# - Sodium: {nutrition.get('sodium')} mg

# Give response in this exact format:
# 1. Health Verdict (Healthy / Unhealthy / Moderate)
# 2. Reason (2-3 lines)
# 3. 4-5 practical suggestions based on goal and disease
# """

#         response = gemini_model.generate_content(prompt)

#         return response.text if response else "No recommendation generated"

#     except Exception as e:
#         print(f"Gemini error: {e}")
#         return "AI recommendation unavailable at the moment"

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), weight: float = Form(...)):
#     try:
#         file_location = f"temp_{file.filename}"
#         with open(file_location, "wb") as f:
#             f.write(await file.read())
#         result = predict_nutrients(file_location, weight)
#         if os.path.exists(file_location):
#             os.remove(file_location)
#         return result
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/recommend")
# async def recommend(
#     calories: float = Form(0.0),
#     protein: float = Form(0.0),
#     carbohydrates: float = Form(0.0),
#     fats: float = Form(0.0),
#     fiber: float = Form(0.0),
#     sugars: float = Form(0.0),
#     sodium: float = Form(0.0),
#     goal: str = Form("maintain"),
#     disease: str = Form(None)
# ):
#     try:
#         nutrition = {
#             "calories": float(calories),
#             "protein": float(protein),
#             "carbohydrates": float(carbohydrates),
#             "fats": float(fats),
#             "fiber": float(fiber),
#             "sugars": float(sugars),
#             "sodium": float(sodium)
#         }
#         ai_response = generate_ai_recommendation(nutrition, goal, disease)
#         return {
#             "recommendations": [ai_response],
#             "goal": goal,
#             "disease": disease
#         }
#     except Exception as e:
#         print(f"Recommend error: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 7860))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)




from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import google.generativeai as genai
import os

app = FastAPI(title="NutriScan AI Backend")

# ==================== CORS (Fixed for your Netlify) ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fyp-project1.netlify.app",   # ← Your frontend
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GEMINI SETUP ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set!")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

print("✅ Gemini configured")

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    try:
        print("📥 Starting model download & load...")
        global nutri_model                  # global bana lo
        nutri_model = download_model()      # ab yeh model return karega
        print("✅ Model ready and loaded!")
    except Exception as e:
        print(f"❌ Critical Model Error: {e}")
        import traceback
        traceback.print_exc()

# ==================== ROUTES ====================
@app.get("/")
async def home():
    return {"message": "Backend is running!", "status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = predict_nutrients(file_location, weight)

        if os.path.exists(file_location):
            os.remove(file_location)

        return result
    except Exception as e:
        print(f"Predict Error: {e}")
        return {"error": str(e)}

@app.post("/recommend")
async def recommend(
    calories: float = Form(0.0),
    protein: float = Form(0.0),
    carbohydrates: float = Form(0.0),
    fats: float = Form(0.0),
    fiber: float = Form(0.0),
    sugars: float = Form(0.0),
    sodium: float = Form(0.0),
    goal: str = Form("maintain"),
    disease: str = Form("none")
):
    try:
        nutrition = {
            "calories": calories, "protein": protein, "carbohydrates": carbohydrates,
            "fats": fats, "fiber": fiber, "sugars": sugars, "sodium": sodium
        }

        prompt = f"""You are a professional nutritionist.
User Goal: {goal}
Disease: {disease or 'None'}

Nutrition:
- Calories: {nutrition['calories']} kcal
- Protein: {nutrition['protein']}g
- Carbs: {nutrition['carbohydrates']}g
- Fats: {nutrition['fats']}g
- Fiber: {nutrition['fiber']}g
- Sugars: {nutrition['sugars']}g
- Sodium: {nutrition['sodium']}mg

Respond in this exact format:
1. Health Verdict (Healthy / Unhealthy / Moderate)
2. Reason (2-3 lines)
3. 4-5 practical suggestions"""

        response = gemini_model.generate_content(prompt)
        return {"recommendations": [response.text]}

    except Exception as e:
        print(f"Recommend Error: {e}")
        return {"error": "Recommendation failed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
