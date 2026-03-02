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


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os

# Initialize FastAPI
app = FastAPI()

# Ensure model is downloaded from Drive
download_model()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Recommendation Logic
# -----------------------------
def generate_recommendation(nutrition, goal, disease):
    recommendations = []

    calories = nutrition.get("calories", 0)
    sugar = nutrition.get("sugars", 0)
    sodium = nutrition.get("sodium", 0)
    fats = nutrition.get("fats", 0)

    # Goal-based
    if goal == "weight_loss":
        if calories > 400:
            recommendations.append("High calories – reduce portion size.")
        if fats > 20:
            recommendations.append("High fat – avoid frequent consumption.")
        recommendations.append("Prefer boiled, grilled or low-oil foods.")

    elif goal == "weight_gain":
        if calories > 300:
            recommendations.append("Good high-energy food for weight gain.")
        else:
            recommendations.append("Add more calorie-dense foods with this meal.")

    elif goal == "maintain":
        recommendations.append("Consume in moderate portion to maintain weight.")

    # Disease-based
    if disease == "diabetes":
        if sugar > 10:
            recommendations.append("High sugar – not recommended for diabetes.")

    if disease == "hypertension":
        if sodium > 400:
            recommendations.append("High sodium – avoid for blood pressure patients.")

    # General
    if calories < 150:
        recommendations.append("Low calorie – healthy light option.")

    return recommendations


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    weight: float = Form(...),
    gender: str = Form(None),
    age: int = Form(None),
    goal: str = Form("maintain"),
    disease: str = Form(None)
):
    # Save file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Nutrition prediction (UNCHANGED)
    result = predict_nutrients(file_location, weight)

    # Remove temp file
    if os.path.exists(file_location):
        os.remove(file_location)

    # -----------------------------
    # Add Recommendation
    # -----------------------------
    recommendations = generate_recommendation(result, goal, disease)

    # Add into response
    result["recommendations"] = recommendations
    result["user_goal"] = goal
    result["disease"] = disease

    return result


# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
