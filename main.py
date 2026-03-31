from fastapi import FastAPI, UploadFile, File, Form

from fastapi.middleware.cors import CORSMiddleware

from predict_nutrient import predict_nutrients

from download_model import download_model

import os

import requests

import time



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

# HF CONFIG

# -----------------------------

HF_TOKEN = "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"



# Primary (DeepSeek)

HF_URL_PRIMARY = "https://router.huggingface.co/hf-inference/models/deepseek-ai/DeepSeek-V3"



# Fallback (Mistral)

HF_URL_FALLBACK = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct"





# -----------------------------

# AI FUNCTION

# -----------------------------

def call_model(url, payload, headers):

    try:

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        data = response.json()



        if isinstance(data, list) and "generated_text" in data[0]:

            return data[0]["generated_text"]



        if isinstance(data, list):

            return str(data[0])



        if isinstance(data, dict):

            if "error" in data:

                return None

            return str(data)



    except:

        return None





def generate_ai_recommendation(nutrition, goal, disease, age, gender):



    prompt = f"""

You are a nutrition expert.



Age: {age}

Gender: {gender}

Goal: {goal}

Disease: {disease}



Calories: {nutrition['calories']}

Protein: {nutrition['protein']}

Carbs: {nutrition['carbohydrates']}

Fats: {nutrition['fats']}

Fiber: {nutrition['fiber']}

Sugar: {nutrition['sugars']}

Sodium: {nutrition['sodium']}



Give 5 line simple advice.

"""



    headers = {

        "Authorization": f"Bearer YOUR_OPENROUTER_KEY"

    }



    payload = {

        "model": "mistralai/mistral-7b-instruct:free",

        "messages": [

            {"role": "user", "content": prompt}

        ]

    }



    try:

        response = requests.post(

            "https://openrouter.ai/api/v1/chat/completions",

            headers=headers,

            json=payload,

            timeout=25

        )



        data = response.json()



        # 🔥 SAFE CHECK (IMPORTANT)

        if "error" in data:

            return f"AI ERROR: {data['error']['message']}"



        if "choices" not in data:

            return f"INVALID RESPONSE: {data}"



        return data["choices"][0]["message"]["content"]



    except Exception as e:

        return f"REQUEST FAILED: {str(e)}"



    prompt = f"""

You are a professional nutrition expert.



User:

Age: {age}

Gender: {gender}

Goal: {goal}

Disease: {disease}



Nutrition:

Calories: {nutrition['calories']}

Protein: {nutrition['protein']}

Carbs: {nutrition['carbohydrates']}

Fats: {nutrition['fats']}

Fiber: {nutrition['fiber']}

Sugar: {nutrition['sugars']}

Sodium: {nutrition['sodium']}



Give clear diet advice in 5 lines.

"""



    headers = {

        "Authorization": f"Bearer {HF_TOKEN}"

    }



    payload = {

        "inputs": prompt,

        "parameters": {

            "max_new_tokens": 200

        }

    }



    # Try DeepSeek first

    for _ in range(2):

        result = call_model(HF_URL_PRIMARY, payload, headers)

        if result:

            return result

        time.sleep(2)



    # Fallback → Mistral

    for _ in range(2):

        result = call_model(HF_URL_FALLBACK, payload, headers)

        if result:

            return result

        time.sleep(2)



    return "AI not responding, try again."





# -----------------------------

# Predict

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

# Recommend

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

@app.get("/")

def home():

    return {"message": "Nutrition AI Running 🚀"} write full code
