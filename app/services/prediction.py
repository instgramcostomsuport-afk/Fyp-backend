import os
import time
import asyncio
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from typing import Dict, Any, List

class NutritionService:
    """Loads nutrition database (or default)"""
    def __init__(self, csv_path=None):
        # Replace with CSV read if you have one
        self.data = {
            "Apple": {"calories": 52, "protein_g": 0.3, "carbs_g": 14, "fat_g": 0.2, "fiber_g": 2.4, "sugars_g": 10, "sodium_mg": 1, "calcium_mg": 6, "iron_mg": 0.1, "vitamin_c_mg": 4.6},
            "Banana": {"calories": 96, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.3, "fiber_g": 2.6, "sugars_g": 14, "sodium_mg": 1, "calcium_mg": 5, "iron_mg": 0.3, "vitamin_c_mg": 8.7},
            "Orange": {"calories": 47, "protein_g": 0.9, "carbs_g": 12, "fat_g": 0.1, "fiber_g": 2.4, "sugars_g": 9, "sodium_mg": 0, "calcium_mg": 40, "iron_mg": 0.1, "vitamin_c_mg": 53.2}
        }

    async def get_nutrition(self, food_name: str, serving_size: int = 100):
        info = self.data.get(food_name)
        if not info:
            return {"error": f"Food '{food_name}' not found in nutrition database"}
        return {k: round(v * serving_size / 100, 2) for k, v in info.items()}

    async def get_database_size(self):
        return len(self.data)


class PredictionService:
    def __init__(self, model_path: str, nutrition_csv_path: str = None):
        self.model_path = model_path
        self.nutrition_service = NutritionService(nutrition_csv_path)
        self.model = None
        self.class_labels = {}  # Will be loaded from model metadata or manually
        self._load_model()
        self._load_class_labels()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        print(f"⚡ Loading model from {self.model_path} ...")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        print("✓ Model loaded successfully")

    def _load_class_labels(self):
        # Replace this with your real class labels if you have a file
        self.class_labels = {0: "Apple", 1: "Banana", 2: "Orange"}

    async def health_check(self) -> Dict[str, Any]:
        test_success = False
        try:
            dummy_image = np.zeros((1, 299, 299, 3))
            _ = self.model.predict(dummy_image)
            test_success = True
        except:
            test_success = False
        return {
            "status": "healthy" if test_success else "unhealthy",
            "model_loaded": self.model is not None,
            "nutrition_calculator_loaded": self.nutrition_service is not None,
            "test_prediction_success": test_success,
            "timestamp": time.time()
        }

    async def predict_food_only(self, image_data, top_predictions: int = 3) -> Dict[str, Any]:
        try:
            # Convert bytes to PIL image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                raise ValueError("Invalid image type")

            # Resize and normalize
            image = image.resize((299, 299))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            preds = self.model.predict(image_array, verbose=0)[0]
            top_idx = np.argsort(preds)[-top_predictions:][::-1]

            results = []
            for rank, idx in enumerate(top_idx):
                results.append({
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": self.class_labels.get(idx, f"class_{idx}"),
                    "confidence": float(preds[idx] * 100)
                })

            return {
                "status": "success",
                "best_prediction": results[0],
                "predictions": results
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "predictions": []}

    async def predict_food_and_nutrition(self, image_data, serving_size_grams: int = 100, top_predictions: int = 3, include_alternatives: bool = True):
        prediction = await self.predict_food_only(image_data, top_predictions)
        best_food = prediction.get("best_prediction", {}).get("class_name", "Unknown")
        nutrition = await self.nutrition_service.get_nutrition(best_food, serving_size_grams)
        return {
            "status": "success",
            "food_prediction": prediction,
            "nutrition_analysis": nutrition,
            "serving_size_grams": serving_size_grams
        }

    async def get_available_foods(self) -> List[str]:
        return list(self.class_labels.values())

    async def get_nutrition_only(self, food_class: str, serving_size: int):
        return await self.nutrition_service.get_nutrition(food_class, serving_size)
