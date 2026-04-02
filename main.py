from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_nutrient import predict_nutrients
from download_model import download_model
import os
import requests

app = FastAPI()

# -------------------------
# CORS FIRST — before everything
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DOWNLOAD MODEL ON STARTUP
# -------------------------
@app.on_event("startup")
def startup_event():
    try:
        print("📥 Starting model download check...")
        download_model()
        print("✅ Model ready")
    except Exception as e:
        print(f"❌ Startup error: {e}")

# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

# -------------------------
# AI RECOMMENDATION
# -------------------------
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

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), weight: float = Form(...)):
    try:
        print("📸 Received file:", file.filename)

        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as f:
            f.write(await file.read())

        print("📊 Running prediction...")
        result = predict_nutrients(file_location, weight)
        print("✅ Result:", result)

        if os.path.exists(file_location):
            os.remove(file_location)

        return result

    except Exception as e:
        print(f"❌ Predict error: {e}")
        return {"error": str(e)}

# -------------------------
# RECOMMEND
# -------------------------
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
        print(f"❌ Recommend error: {e}")
        return {"error": str(e)}

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
```

Click **Commit changes**.

---

**Step 3 — Check `requirements.txt`**

Open it on GitHub and make sure these are all there:
```
fastapi
uvicorn
tensorflow
numpy
pandas
pillow
gdown
requests
python-multipart
```

If `gdown` is missing — add it. Commit.

---

**Step 4 — Redeploy on Railway**

Railway auto-deploys when you push to GitHub. Go to Railway → your service → **Deployments** → watch the logs.

You should see:
```
📥 Downloading model from Google Drive...
✅ Model downloaded — 110.x MB
✅ Model ready
```

If you see anything else — copy those logs and paste here immediately.

---

**Step 5 — Paste your frontend API call**

Open your frontend `src` folder — find where you call the backend. Paste that code here so I can verify the URL is correct.

What is your Railway backend URL? It looks like:
```
https://something.railway.app
