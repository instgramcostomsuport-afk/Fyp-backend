# # # from fastapi import FastAPI, UploadFile, File, Form
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from predict_nutrient import predict_nutrients
# # # from download_model import download_model
# # # import os

# # # app = FastAPI()

# # # # Ensure model is downloaded
# # # download_model()

# # # # Enable CORS
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],  # In production, replace with your frontend URL
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # -----------------------------
# # # # Recommendation Logic
# # # # -----------------------------
# # # def generate_recommendation(nutrition, goal, disease):
# # #     recommendations = []

# # #     calories = nutrition.get("calories", 0)
# # #     sugar = nutrition.get("sugars", 0)
# # #     sodium = nutrition.get("sodium", 0)
# # #     fats = nutrition.get("fats", 0)

# # #     # Goal-based
# # #     if goal == "weight_loss":
# # #         if calories > 400:
# # #             recommendations.append("High calories – reduce portion size.")
# # #         if fats > 20:
# # #             recommendations.append("High fat – avoid frequent consumption.")
# # #         recommendations.append("Prefer boiled, grilled or low-oil foods.")

# # #     elif goal == "weight_gain":
# # #         if calories > 300:
# # #             recommendations.append("Good high-energy food for weight gain.")
# # #         else:
# # #             recommendations.append("Add more calorie-dense foods with this meal.")

# # #     elif goal == "maintain":
# # #         recommendations.append("Consume in moderate portion to maintain weight.")

# # #     # Disease-based
# # #     if disease == "diabetes":
# # #         if sugar > 10:
# # #             recommendations.append("High sugar – not recommended for diabetes.")
# # #     if disease == "hypertension":
# # #         if sodium > 400:
# # #             recommendations.append("High sodium – avoid for blood pressure patients.")

# # #     # General
# # #     if calories < 150:
# # #         recommendations.append("Low calorie – healthy light option.")

# # #     return recommendations

# # # # -----------------------------
# # # # Nutrition Prediction Endpoint
# # # # -----------------------------
# # # @app.post("/predict")
# # # async def predict(file: UploadFile = File(...), weight: float = Form(...)):
# # #     # Save file temporarily
# # #     file_location = f"temp_{file.filename}"
# # #     with open(file_location, "wb") as f:
# # #         f.write(await file.read())

# # #     # Predict nutrition
# # #     result = predict_nutrients(file_location, weight)

# # #     # Remove temp file
# # #     if os.path.exists(file_location):
# # #         os.remove(file_location)

# # #     # Return nutrition ONLY, frontend will handle user input for recommendation
# # #     return result

# # # # -----------------------------
# # # # Recommendation Endpoint
# # # # -----------------------------
# # # @app.post("/recommend")
# # # async def recommend(
# # #     calories: float = Form(0.0),
# # #     protein: float = Form(0.0),
# # #     carbohydrates: float = Form(0.0),
# # #     fats: float = Form(0.0),
# # #     fiber: float = Form(0.0),
# # #     sugars: float = Form(0.0),
# # #     sodium: float = Form(0.0),
# # #     goal: str = Form("maintain"),
# # #     disease: str = Form(None)
# # # ):
# # #     try:
# # #         nutrition = {
# # #             "calories": float(calories),
# # #             "protein": float(protein),
# # #             "carbohydrates": float(carbohydrates),
# # #             "fats": float(fats),
# # #             "fiber": float(fiber),
# # #             "sugars": float(sugars),
# # #             "sodium": float(sodium)
# # #         }
# # #         recs = generate_recommendation(nutrition, goal, disease)
# # #         return {"recommendations": recs, "goal": goal, "disease": disease}
# # #     except Exception as e:
# # #         return {"error": str(e)}

# # # # Run server
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     port = int(os.environ.get("PORT", 8000))
# # #     uvicorn.run("main:app", host="0.0.0.0", port=port)

# # from fastapi import FastAPI, UploadFile, File, Form
# # from fastapi.middleware.cors import CORSMiddleware
# # from predict_nutrient import predict_nutrients
# # from download_model import download_model
# # import os
# # import requests

# # app = FastAPI()

# # # Ensure model is downloaded
# # download_model()

# # # Enable CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # change in production
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import os
# import requests

# # -----------------------------
# # Initialize FastAPI
# # -----------------------------
# app = FastAPI()

# # -----------------------------
# # Ensure model is downloaded
# # -----------------------------
# download_model()

# # -----------------------------
# # Enable CORS
# # -----------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # replace with frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # Debug endpoint to check environment variable
# # -----------------------------
# @app.get("/env")
# async def check_env():
#     return {"OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY")}

# # -----------------------------
# # AI Recommendation via Qwen with fallback
# # -----------------------------
# def generate_ai_recommendation(nutrition: dict, goal: str, disease: str):
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         raise Exception("OPENROUTER_API_KEY not set!")

#     prompt = f"""
# You are a professional nutritionist AI.
# User Goal: {goal}
# Disease: {disease}
# Nutrition Values:
# Calories: {nutrition.get('calories')}
# Protein: {nutrition.get('protein')}
# Carbohydrates: {nutrition.get('carbohydrates')}
# Fats: {nutrition.get('fats')}
# Fiber: {nutrition.get('fiber')}
# Sugars: {nutrition.get('sugars')}
# Sodium: {nutrition.get('sodium')}

# Give response in this format:
# 1. Health Verdict (Healthy / Unhealthy)
# 2. Reason
# 3. 3-5 practical suggestions
# """

#     models_to_try = ["qwen/qwen-7b-chat", "qwen/qwen-7b-mini"]  # fallback to mini if chat fails
#     last_error = None

#     for model_name in models_to_try:
#         try:
#             response = requests.post(
#                 "https://openrouter.ai/api/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {api_key}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": model_name,
#                     "messages": [{"role": "user", "content": prompt}]
#                 },
#                 timeout=60
#             )

#             # Log status and body for debugging
#             print(f"Trying model {model_name}")
#             print("Status code:", response.status_code)
#             print("Response body:", response.text)

#             if response.status_code == 200:
#                 return response.json()["choices"][0]["message"]["content"]
#             else:
#                 last_error = f"{model_name} failed with {response.status_code}: {response.text}"
#         except Exception as e:
#             last_error = f"{model_name} request exception: {str(e)}"

#     # If all models fail
#     raise Exception(f"All models failed. Last error: {last_error}")

# # -----------------------------
# # Nutrition Prediction Endpoint
# # -----------------------------
# @app.post("/predict")
# async def predict(file: UploadFile = File(...), weight: float = Form(...)):
#     file_location = f"temp_{file.filename}"
#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     try:
#         result = predict_nutrients(file_location, weight)
#     finally:
#         if os.path.exists(file_location):
#             os.remove(file_location)

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
#         ai_response = generate_ai_recommendation(nutrition, goal, disease)
#         return {
#             "recommendations": [ai_response],
#             "goal": goal,
#             "disease": disease
#         }
#     except Exception as e:
#         # Return full error for debugging
#         return {"error": str(e)}

# # -----------------------------
# # Run server
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)


# main.py
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from predict_nutrient import predict_nutrients
# from download_model import download_model
# import os
# import google.generativeai as genai

# app = FastAPI()

# # -----------------------------
# # Load Model
# # -----------------------------
# download_model()

# # -----------------------------
# # CORS
# # -----------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # 🔥 GEMINI CONFIG
# # -----------------------------
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("gemini-2.0-flash")


# # -----------------------------
# # AI FUNCTION
# # -----------------------------
# def generate_ai_recommendation(nutrition, goal, disease, age, gender):

#     prompt = f"""
# You are a professional nutrition expert.

# User Info:
# Age: {age}
# Gender: {gender}
# Goal: {goal}
# Disease: {disease}

# Nutrition:
# Calories: {nutrition['calories']}
# Protein: {nutrition['protein']}
# Carbohydrates: {nutrition['carbohydrates']}
# Fats: {nutrition['fats']}
# Fiber: {nutrition['fiber']}
# Sugars: {nutrition['sugars']}
# Sodium: {nutrition['sodium']}

# Give response in this format:
# 1. Health Verdict
# 2. Reason
# 3. 3 practical suggestions
# """

#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"AI Error: {str(e)}"


# # -----------------------------
# # Predict Endpoint
# # -----------------------------



# # -----------------------------
# # Recommendation Endpoint
# # -----------------------------
# @app.post("/recommend")
# async def recommend(
#     calories: float = Form(...),
#     protein: float = Form(...),
#     carbohydrates: float = Form(...),
#     fats: float = Form(...),
#     fiber: float = Form(...),
#     sugars: float = Form(...),
#     sodium: float = Form(...),

#     age: int = Form(0),
#     gender: str = Form("unknown"),
#     goal: str = Form("maintain"),
#     disease: str = Form("")
# ):

#     nutrition = {
#         "calories": calories,
#         "protein": protein,
#         "carbohydrates": carbohydrates,
#         "fats": fats,
#         "fiber": fiber,
#         "sugars": sugars,
#         "sodium": sodium
#     }

#     ai_response = generate_ai_recommendation(
#         nutrition,
#         goal,
#         disease,
#         age,
#         gender
#     )

#     return {
#         "recommendations": [ai_response],
#         "status": "success"
#     }


# # -----------------------------
# # Health Check
# # -----------------------------
# @app.get("/")
# def home():
#     return {"message": "Nutrition AI Running 🚀"}


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os
import google.generativeai as genai


app = FastAPI()

download_model()
# -------------------------
# CORS (FIXED)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):
    # Save file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Predict nutrition
    result = predict_nutrients(file_location, weight)

    # Remove temp file
    if os.path.exists(file_location):
        os.remove(file_location)

# # #     # Return nutrition ONLY, frontend will handle user input for recommendation
    return result
# -------------------------
# GEMINI SETUP (SAFE)
# -------------------------
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


# -------------------------
# HOME
# -------------------------



# -------------------------
# RECOMMENDATION (FIXED GEMINI)
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

User Details:
Age: {age}
Gender: {gender}
Goal: {goal}
Disease: {disease}

Food Nutrition:
Calories: {calories}
Protein: {protein}
Carbs: {carbohydrates}
Fats: {fats}
Fiber: {fiber}
Sugar: {sugars}
Sodium: {sodium}

Give a short 4-6 line personalized diet recommendation.
Be practical and simple.
"""

        response = model.generate_content(prompt)

        # IMPORTANT: Gemini correct output
        text = response.text if response else "No response"

        return {
            "recommendations": [text]
        }

    except Exception as e:
        print("🔥 GEMINI ERROR:", str(e))

        return {
            "error": str(e),
            "recommendations": [
                "AI temporarily failed. Check API key or try again."
            ]
        }


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
