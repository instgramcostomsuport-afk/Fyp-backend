from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os

# Initialize FastAPI
app = FastAPI()

# Ensure the model exists (download if not)
download_model()  # This should check if .h5 exists, otherwise download

# Enable CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):
    # Save uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Call your prediction function
    result = predict_nutrients(file_location, weight)
    
    # Clean up temporary file
    if os.path.exists(file_location):
        os.remove(file_location)
    
    return result

# Run server when executed directly (Render will use this)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render sets $PORT
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
