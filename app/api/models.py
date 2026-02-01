# app/api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    very_low = "very_low"

class Prediction(BaseModel):
    rank: int = Field(..., description="Ranking of this prediction")
    class_id: int = Field(..., description="Numerical class ID")
    class_name: str = Field(..., description="Food class name")
    confidence: float = Field(..., description="Confidence percentage (0-100)")

class PredictionResponse(BaseModel):
    status: str = Field(..., description="Response status")
    predictions: List[Prediction] = Field(..., description="List of predictions")
    best_prediction: Optional[Prediction] = Field(None, description="Best prediction")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class Macronutrients(BaseModel):
    protein_g: float = Field(..., description="Protein in grams")
    fat_g: float = Field(..., description="Fat in grams")
    carbs_g: float = Field(..., description="Carbohydrates in grams")
    fiber_g: float = Field(..., description="Fiber in grams")
    sugars_g: float = Field(..., description="Sugars in grams")

class Micronutrients(BaseModel):
    sodium_mg: float = Field(..., description="Sodium in milligrams")
    calcium_mg: Optional[float] = Field(None, description="Calcium in milligrams")
    iron_mg: Optional[float] = Field(None, description="Iron in milligrams")
    vitamin_c_mg: Optional[float] = Field(None, description="Vitamin C in milligrams")

class CalorieBreakdown(BaseModel):
    from_protein: float = Field(..., description="Calories from protein")
    from_fat: float = Field(..., description="Calories from fat")
    from_carbs: float = Field(..., description="Calories from carbohydrates")
    total_calculated: float = Field(..., description="Total calculated calories")

class NutritionInfo(BaseModel):
    food_item: str = Field(..., description="Food item name")
    serving_size_grams: int = Field(..., description="Serving size in grams")
    calories: float = Field(..., description="Total calories")
    macronutrients: Macronutrients = Field(..., description="Macronutrient information")
    micronutrients: Micronutrients = Field(..., description="Micronutrient information")
    calorie_breakdown: CalorieBreakdown = Field(..., description="Calorie breakdown by macronutrients")
    data_source: str = Field(..., description="Source of nutrition data (database/estimated)")

class ConfidenceAssessment(BaseModel):
    level: ConfidenceLevel = Field(..., description="Confidence level")
    description: str = Field(..., description="Human-readable description")
    reliability: str = Field(..., description="Reliability assessment")

class ServingInfo(BaseModel):
    serving_size_grams: int = Field(..., description="Serving size in grams")
    serving_description: str = Field(..., description="Human-readable serving description")

class ImageValidation(BaseModel):
    valid: bool = Field(..., description="Whether image is valid")
    format: Optional[str] = Field(None, description="Image format")
    mode: Optional[str] = Field(None, description="Image mode")
    size: Optional[List[int]] = Field(None, description="Image dimensions")
    errors: List[str] = Field(default=[], description="Validation errors")

class FoodPrediction(BaseModel):
    primary: Prediction = Field(..., description="Primary prediction")
    alternatives: List[Prediction] = Field(default=[], description="Alternative predictions")

class NutritionAnalysis(BaseModel):
    primary: NutritionInfo = Field(..., description="Primary nutrition information")
    alternatives: List[Dict[str, Any]] = Field(default=[], description="Alternative nutrition information")

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="Response status")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_validation: ImageValidation = Field(..., description="Image validation results")
    food_prediction: FoodPrediction = Field(..., description="Food prediction results")
    nutrition_analysis: NutritionAnalysis = Field(..., description="Nutrition analysis results")
    serving_info: ServingInfo = Field(..., description="Serving information")
    confidence_assessment: ConfidenceAssessment = Field(..., description="Confidence assessment")

class NutritionResponse(BaseModel):
    food_item: str = Field(..., description="Food item name")
    serving_size_grams: int = Field(..., description="Serving size in grams")
    calories: float = Field(..., description="Total calories")
    macronutrients: Macronutrients = Field(..., description="Macronutrient information")
    micronutrients: Micronutrients = Field(..., description="Micronutrient information")
    calorie_breakdown: CalorieBreakdown = Field(..., description="Calorie breakdown")
    data_source: str = Field(..., description="Data source")

class NutritionRequest(BaseModel):
    food_class: str = Field(..., description="Food class name", example="apple_pie")
    serving_size: int = Field(100, description="Serving size in grams", ge=1, le=2000)

class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    top_predictions: int = Field(5, description="Number of top predictions", ge=1, le=10)
    serving_size: int = Field(100, description="Serving size in grams", ge=1, le=2000)
    include_alternatives: bool = Field(True, description="Include alternative predictions")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    nutrition_calculator_loaded: bool = Field(..., description="Whether nutrition calculator is loaded")
    test_prediction_success: bool = Field(..., description="Whether test prediction succeeded")
    timestamp: float = Field(..., description="Timestamp of health check")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    input_shape: List[int] = Field(..., description="Model input shape")
    output_shape: List[int] = Field(..., description="Model output shape")
    num_classes: int = Field(..., description="Number of classes")
    total_parameters: int = Field(..., description="Total model parameters")

class ServiceInfoResponse(BaseModel):
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_info: ModelInfo = Field(..., description="Model information")
    nutrition_database_size: int = Field(..., description="Number of foods in nutrition database")
    supported_image_formats: List[str] = Field(..., description="Supported image formats")
    max_image_size: str = Field(..., description="Maximum image size")
    expected_input_size: str = Field(..., description="Expected input size")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    status_code: Optional[int] = Field(None, description="HTTP status code")
