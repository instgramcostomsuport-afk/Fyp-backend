# app/core/config.py
import os
from pathlib import Path
from typing import Optional

class Settings:
    """Application settings"""
    
    # Project info
    PROJECT_NAME: str = "NutriFoodNet API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered food recognition and nutrition estimation API"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model settings
    MODEL_PATH: Path = MODELS_DIR / "nutrifoodnet_final.h5"
    CLASS_LABELS_PATH: Path = MODELS_DIR / "class_labels.json"
    MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
    NUTRITION_CSV_PATH: Path = DATA_DIR / "nutrition.csv"
    
    # Image processing settings
    MAX_IMAGE_SIZE: int = 4096  # pixels
    MIN_IMAGE_SIZE: int = 32    # pixels
    TARGET_IMAGE_SIZE: tuple = (299, 299)
    MAX_BATCH_SIZE: int = 10
    
    # API limits
    MAX_SERVING_SIZE: int = 2000  # grams
    MIN_SERVING_SIZE: int = 1     # grams
    MAX_TOP_PREDICTIONS: int = 10
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  # Configure for production
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Security (for production)
    SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    def __init__(self):
        """Initialize settings and create directories if needed"""
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
    
    @property
    def model_files_exist(self) -> bool:
        """Check if required model files exist"""
        return (
            self.MODEL_PATH.exists() and
            self.CLASS_LABELS_PATH.exists()
        )
    
    @property
    def nutrition_file_exists(self) -> bool:
        """Check if nutrition CSV exists"""
        return self.NUTRITION_CSV_PATH.exists()

# Create global settings instance
settings = Settings()
    