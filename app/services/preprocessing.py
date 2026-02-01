# app/services/preprocessing.py
import numpy as np
from PIL import Image, ImageOps
import io
import base64
from typing import Union, Tuple, Dict, Any

class ImagePreprocessor:
    """
    Image preprocessing service for NutriFoodNet
    """
    
    def __init__(self, target_size: Tuple[int, int] = (299, 299)):
        self.target_size = target_size
    
    def preprocess_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image_input: Image in various formats
            
        Returns:
            np.ndarray: Preprocessed image ready for prediction
        """
        try:
            # Load image
            image = self._load_image(image_input)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def _load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various input formats"""
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input)
        elif isinstance(image_input, bytes):
            # Bytes data
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            if image_input.max() <= 1.0:
                image_input = (image_input * 255).astype(np.uint8)
            return Image.fromarray(image_input)
        elif hasattr(image_input, 'read'):
            # File-like object
            return Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def validate_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Validate if image is suitable for processing
        
        Args:
            image_input: Image to validate
            
        Returns:
            dict: Validation results
        """
        try:
            image = self._load_image(image_input)
            
            validation = {
                'valid': True,
                'format': getattr(image, 'format', 'Unknown'),
                'mode': image.mode,
                'size': list(image.size),
                'errors': []
            }
            
            # Check minimum size
            if image.size[0] < 32 or image.size[1] < 32:
                validation['errors'].append('Image too small (minimum 32x32 pixels)')
            
            # Check maximum size
            if image.size[0] > 4096 or image.size[1] > 4096:
                validation['errors'].append('Image too large (maximum 4096x4096 pixels)')
            
            # Check image mode
            if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                validation['errors'].append(f'Unsupported image mode: {image.mode}')
            
            # Check if image has content
            if image.size[0] == 0 or image.size[1] == 0:
                validation['errors'].append('Image has zero dimensions')
            
            if validation['errors']:
                validation['valid'] = False
                
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Cannot load image: {str(e)}']
            }
    
    def preprocess_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Preprocess image from base64 string
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Process as bytes
            return self.preprocess_image(image_bytes)
            
        except Exception as e:
            raise ValueError(f"Error processing base64 image: {str(e)}")
    
    def batch_preprocess_images(self, image_inputs: list) -> np.ndarray:
        """
        Preprocess multiple images at once
        
        Args:
            image_inputs: List of image inputs
            
        Returns:
            np.ndarray: Batch of preprocessed images
        """
        batch_images = []
        
        for image_input in image_inputs:
            try:
                processed = self.preprocess_image(image_input)
                batch_images.append(processed[0])  # Remove batch dimension
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if batch_images:
            return np.array(batch_images)
        else:
            raise ValueError("No valid images found in batch")
    
    def get_image_info(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Get detailed information about an image"""
        try:
            image = self._load_image(image_input)
            
            return {
                'format': getattr(image, 'format', 'Unknown'),
                'mode': image.mode,
                'size': list(image.size),
                'has_transparency': image.mode in ['RGBA', 'LA'] or 'transparency' in image.info,
                'is_animated': getattr(image, 'is_animated', False),
                'n_frames': getattr(image, 'n_frames', 1)
            }
            
        except Exception as e:
            return {'error': str(e)}

# Utility functions for common use cases
def quick_preprocess(image_input, target_size: Tuple[int, int] = (299, 299)) -> np.ndarray:
    """Quick preprocessing function"""
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.preprocess_image(image_input)

def validate_image_file(image_input) -> bool:
    """Quick image validation"""
    preprocessor = ImagePreprocessor()
    validation = preprocessor.validate_image(image_input)
    return validation['valid']
