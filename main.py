# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import os
# import requests

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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
#     prompt = f"You are a professional nutritionist AI. User Goal: {goal}. Disease: {disease}. Nutrition: Calories: {nutrition.get('calories')}, Protein: {nutrition.get('protein')}, Carbohydrates: {nutrition.get('carbohydrates')}, Fats: {nutrition.get('fats')}, Fiber: {nutrition.get('fiber')}, Sugars: {nutrition.get('sugars')}, Sodium: {nutrition.get('sodium')}. Give response: 1. Health Verdict 2. Reason 3. 3-5 suggestions"
#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
#             "Content-Type": "application/json"
#         },
#         json={
#             "model": "qwen/qwen-7b-chat",
#             "messages": [{"role": "user", "content": prompt}]
#         }
#     )
#     return response.json()["choices"][0]["message"]["content"]

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
#         return {"recommendations": [ai_response], "goal": goal, "disease": disease}
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)



from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import google.generativeai as genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

@app.on_event("startup")
def startup_event():
    try:
        print("Downloading model check...")
        download_model()
        print("Model ready")
    except Exception as e:
        print(f"Startup error: {e}")

@app.get("/")
def home():
    return {"message": "Backend running"}

def generate_ai_recommendation(nutrition, goal, disease):
    prompt = f"""
You are a professional nutritionist AI.

User Goal: {goal}
Disease/Condition: {disease}

Nutrition Values Detected:
- Calories: {nutrition.get('calories')} kcal
- Protein: {nutrition.get('protein')} g
- Carbohydrates: {nutrition.get('carbohydrates')} g
- Fats: {nutrition.get('fats')} g
- Fiber: {nutrition.get('fiber')} g
- Sugars: {nutrition.get('sugars')} g
- Sodium: {nutrition.get('sodium')} mg

Give response in this exact format:
1. Health Verdict (Healthy / Unhealthy / Moderate)
2. Reason (2-3 lines)
3. 4-5 practical suggestions based on goal and disease
"""
    response = gemini_model.generate_content(prompt)
    return response.text

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
    disease: str = Form(None)
):
    try:
        nutrition = {
            "calories": float(calories),
            "protein": float(protein),
            "carbohydrates": float(carbohydrates),
            "fats": float(fats),
            "fiber": float(fiber),
            "sugars": float(sugars),
            "sodium": float(sodium)
        }
        ai_response = generate_ai_recommendation(nutrition, goal, disease)
        return {
            "recommendations": [ai_response],
            "goal": goal,
            "disease": disease
        }
    except Exception as e:
        print(f"Recommend error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
