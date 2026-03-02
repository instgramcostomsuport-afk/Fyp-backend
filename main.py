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

# --- New Imports ---
from recommendation import calculate_bmr, get_recommendations

app = FastAPI()

download_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_with_recs")
async def predict_with_recs(
    file: UploadFile = File(...),
    weight: float = Form(...),
    height: float = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    goal: str = Form(...)
):
    # save file
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    result = predict_nutrients(file_location, weight)

    if os.path.exists(file_location):
        os.remove(file_location)

    # Personal calorie need
    bmr = calculate_bmr(weight, height, age, gender)

    # Add recommendation
    recs = get_recommendations(goal)

    return {
        "food_label": result.get("label"),
        "confidence": result.get("confidence"),
        "nutrition": result,
        "daily_calorie_need": round(bmr,2),
        "recommendations": recs
    }
