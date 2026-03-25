# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import os

# app = FastAPI()

# # Ensure model is downloaded
# download_model()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # Recommendation Logic
# # -----------------------------
# def generate_recommendation(nutrition, goal, disease):
#     recommendations = []

#     calories = nutrition.get("calories", 0)
#     sugar = nutrition.get("sugars", 0)
#     sodium = nutrition.get("sodium", 0)
#     fats = nutrition.get("fats", 0)

#     # Goal-based
#     if goal == "weight_loss":
#         if calories > 400:
#             recommendations.append("High calories – reduce portion size.")
#         if fats > 20:
#             recommendations.append("High fat – avoid frequent consumption.")
#         recommendations.append("Prefer boiled, grilled or low-oil foods.")

#     elif goal == "weight_gain":
#         if calories > 300:
#             recommendations.append("Good high-energy food for weight gain.")
#         else:
#             recommendations.append("Add more calorie-dense foods with this meal.")

#     elif goal == "maintain":
#         recommendations.append("Consume in moderate portion to maintain weight.")

#     # Disease-based
#     if disease == "diabetes":
#         if sugar > 10:
#             recommendations.append("High sugar – not recommended for diabetes.")
#     if disease == "hypertension":
#         if sodium > 400:
#             recommendations.append("High sodium – avoid for blood pressure patients.")

#     # General
#     if calories < 150:
#         recommendations.append("Low calorie – healthy light option.")

#     return recommendations

# # -----------------------------
# # Nutrition Prediction Endpoint
# # -----------------------------
# @app.post("/predict")
# async def predict(file: UploadFile = File(...), weight: float = Form(...)):
#     # Save file temporarily
#     file_location = f"temp_{file.filename}"
#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     # Predict nutrition
#     result = predict_nutrients(file_location, weight)

#     # Remove temp file
#     if os.path.exists(file_location):
#         os.remove(file_location)

#     # Return nutrition ONLY, frontend will handle user input for recommendation
#     return result

# # -----------------------------
# # Recommendation Endpoint
# # -----------------------------
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
#         recs = generate_recommendation(nutrition, goal, disease)
#         return {"recommendations": recs, "goal": goal, "disease": disease}
#     except Exception as e:
#         return {"error": str(e)}

# # Run server
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os
import requests

app = FastAPI()

# Ensure model is downloaded
download_model()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 🔥 AI Recommendation (Qwen via API)
# -----------------------------
def generate_ai_recommendation(nutrition, goal, disease):

    prompt = f"""
You are a professional nutritionist AI.

User Goal: {goal}
Disease: {disease}

Nutrition Values:
Calories: {nutrition.get('calories')}
Protein: {nutrition.get('protein')}
Carbohydrates: {nutrition.get('carbohydrates')}
Fats: {nutrition.get('fats')}
Fiber: {nutrition.get('fiber')}
Sugars: {nutrition.get('sugars')}
Sodium: {nutrition.get('sodium')}

Give response in this format:
1. Health Verdict (Healthy / Unhealthy)
2. Reason
3. 3-5 practical suggestions
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen/qwen-7b-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# Nutrition Prediction Endpoint
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
# 🔥 Recommendation Endpoint (UPDATED)
# -----------------------------
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
        return {"error": str(e)}


# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
