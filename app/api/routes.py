# app/api/routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import base64
import io
from typing import Optional, List
import asyncio

from app.api.models import (
    PredictionResponse,
    AnalysisResponse, 
    NutritionResponse,
    NutritionRequest,
    Base64ImageRequest,
    HealthResponse,
    ServiceInfoResponse
)
from app.core.dependencies import get_prediction_service

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(service=Depends(get_prediction_service)):
    """Health check endpoint"""
    try:
        health_status = await service.health_check()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/info", response_model=ServiceInfoResponse)
async def get_service_info(service=Depends(get_prediction_service)):
    """Get service information"""
    try:
        info = await service.get_service_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def predict_food(
    image: UploadFile = File(...),
    top_predictions: int = Form(5),
    service=Depends(get_prediction_service)
):
    """
    Predict food class from uploaded image
    
    - **image**: Image file (JPEG, PNG, etc.)
    - **top_predictions**: Number of top predictions to return (default: 5)
    """
    # Validate image file
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Make prediction
        result = await service.predict_food_only(
            image_data, 
            top_predictions=top_predictions
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/base64", response_model=PredictionResponse)
async def predict_food_base64(
    request: Base64ImageRequest,
    service=Depends(get_prediction_service)
):
    """
    Predict food class from base64 encoded image
    
    - **image_base64**: Base64 encoded image data
    - **top_predictions**: Number of top predictions to return (default: 5)
    """
    try:
        # Decode base64 image
        if request.image_base64.startswith('data:image'):
            base64_data = request.image_base64.split(',')[1]
        else:
            base64_data = request.image_base64
        
        image_bytes = base64.b64decode(base64_data)
        
        # Make prediction
        result = await service.predict_food_only(
            image_bytes,
            top_predictions=request.top_predictions
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_food(
    image: UploadFile = File(...),
    serving_size: int = Form(100),
    top_predictions: int = Form(3),
    include_alternatives: bool = Form(True),
    service=Depends(get_prediction_service)
):
    """
    Complete food analysis: prediction + nutrition estimation
    
    - **image**: Image file (JPEG, PNG, etc.)
    - **serving_size**: Serving size in grams (default: 100)
    - **top_predictions**: Number of predictions to consider (default: 3)
    - **include_alternatives**: Include nutrition for alternative predictions (default: true)
    """
    # Validate inputs
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    if serving_size <= 0 or serving_size > 2000:
        raise HTTPException(
            status_code=400,
            detail="Serving size must be between 1 and 2000 grams"
        )
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Perform complete analysis
        result = await service.predict_food_and_nutrition(
            image_data,
            serving_size_grams=serving_size,
            top_predictions=top_predictions,
            include_alternatives=include_alternatives
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/base64", response_model=AnalysisResponse)
async def analyze_food_base64(
    request: Base64ImageRequest,
    service=Depends(get_prediction_service)
):
    """
    Complete food analysis from base64 encoded image
    """
    # Validate serving size
    if request.serving_size <= 0 or request.serving_size > 2000:
        raise HTTPException(
            status_code=400,
            detail="Serving size must be between 1 and 2000 grams"
        )
    
    try:
        # Decode base64 image
        if request.image_base64.startswith('data:image'):
            base64_data = request.image_base64.split(',')[1]
        else:
            base64_data = request.image_base64
        
        image_bytes = base64.b64decode(base64_data)
        
        # Perform complete analysis
        result = await service.predict_food_and_nutrition(
            image_bytes,
            serving_size_grams=request.serving_size,
            top_predictions=request.top_predictions,
            include_alternatives=request.include_alternatives
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/nutrition", response_model=NutritionResponse)
async def get_nutrition(
    request: NutritionRequest,
    service=Depends(get_prediction_service)
):
    """
    Get nutrition information for a known food class
    
    - **food_class**: Name of the food class (e.g., "apple_pie")
    - **serving_size**: Serving size in grams (default: 100)
    """
    # Validate serving size
    if request.serving_size <= 0 or request.serving_size > 2000:
        raise HTTPException(
            status_code=400,
            detail="Serving size must be between 1 and 2000 grams"
        )
    
    try:
        result = await service.get_nutrition_only(
            request.food_class,
            request.serving_size
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nutrition lookup failed: {str(e)}")

@router.post("/batch/analyze")
async def batch_analyze_images(
    images: List[UploadFile] = File(...),
    serving_size: int = Form(100),
    service=Depends(get_prediction_service)
):
    """
    Analyze multiple images at once
    
    - **images**: List of image files
    - **serving_size**: Serving size in grams for all images (default: 100)
    """
    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    # Validate serving size
    if serving_size <= 0 or serving_size > 2000:
        raise HTTPException(
            status_code=400,
            detail="Serving size must be between 1 and 2000 grams"
        )
    
    try:
        # Read all image data
        image_data_list = []
        for image in images:
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {image.filename} must be an image"
                )
            
            data = await image.read()
            image_data_list.append(data)
        
        # Process batch
        results = await service.batch_analyze_images(
            image_data_list,
            serving_size_grams=serving_size
        )
        
        return {
            "status": "success",
            "total_images": len(images),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/foods/available")
async def get_available_foods(service=Depends(get_prediction_service)):
    """Get list of available food classes"""
    try:
        foods = await service.get_available_foods()
        return {
            "total_foods": len(foods),
            "foods": foods
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get food list: {str(e)}")
