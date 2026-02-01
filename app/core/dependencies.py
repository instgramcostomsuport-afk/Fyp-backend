import os
from app.services.prediction import PredictionService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "nutrifoodnet_final.h5")

_prediction_service = None

def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(
            model_path=MODEL_PATH
        )
    return _prediction_service
