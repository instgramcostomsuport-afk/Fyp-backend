# app/services/nutrition.py
import pandas as pd
import os
from typing import Dict, Any, List
import asyncio

class NutritionService:
    """
    Service for nutrition calculation based on your nutrition.csv format
    """
    
    def __init__(self, nutrition_csv_path: str):
        self.nutrition_csv_path = nutrition_csv_path
        self.nutrition_df = None
        self._load_nutrition_data()
    
    def _load_nutrition_data(self):
        """Load nutrition data from CSV"""
        if not os.path.exists(self.nutrition_csv_path):
            print(f"Warning: Nutrition CSV not found at {self.nutrition_csv_path}")
            self._create_sample_data()
            return
        
        try:
            self.nutrition_df = pd.read_csv(self.nutrition_csv_path)
            print(f"✓ Nutrition data loaded: {len(self.nutrition_df)} records")
            
            # Validate required columns
            required_columns = ['label', 'weight', 'calories', 'protein', 'carbohydrates', 'fats']
            missing_columns = [col for col in required_columns if col not in self.nutrition_df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in nutrition data: {missing_columns}")
                self._create_sample_data()
            
        except Exception as e:
            print(f"Error loading nutrition data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample nutrition data as fallback"""
        sample_data = [
            {'label': 'apple_pie', 'weight': 100, 'calories': 300, 'protein': 3, 'carbohydrates': 45, 'fats': 12, 'fiber': 2, 'sugars': 20, 'sodium': 150},
            {'label': 'pizza', 'weight': 100, 'calories': 266, 'protein': 11, 'carbohydrates': 33, 'fats': 10, 'fiber': 2, 'sugars': 4, 'sodium': 640},
            {'label': 'hamburger', 'weight': 100, 'calories': 295, 'protein': 17, 'carbohydrates': 25, 'fats': 14, 'fiber': 2, 'sugars': 4, 'sodium': 500},
            {'label': 'sushi', 'weight': 100, 'calories': 150, 'protein': 8, 'carbohydrates': 28, 'fats': 2, 'fiber': 1, 'sugars': 3, 'sodium': 400},
            {'label': 'baby_back_ribs', 'weight': 100, 'calories': 292, 'protein': 19, 'carbohydrates': 0, 'fats': 10, 'fiber': 0, 'sugars': 0, 'sodium': 500},
        ]
        
        self.nutrition_df = pd.DataFrame(sample_data)
        print("✓ Using sample nutrition data")
    
    async def get_nutrition(self, food_class: str, serving_size_grams: int) -> Dict[str, Any]:
        """
        Get nutrition information for a food class and serving size
        
        Args:
            food_class: Food class name (e.g., "apple_pie")
            serving_size_grams: Desired serving size in grams
            
        Returns:
            dict: Nutrition information
        """
        try:
            # Find nutrition data for this food class
            food_data = await asyncio.to_thread(self._find_food_data, food_class, serving_size_grams)
            
            if food_data is None:
                return self._get_estimated_nutrition(food_class, serving_size_grams)
            
            return self._format_nutrition_response(food_class, serving_size_grams, food_data)
            
        except Exception as e:
            return {
                'error': str(e),
                'food_item': food_class,
                'serving_size_grams': serving_size_grams
            }
    
    def _find_food_data(self, food_class: str, serving_size_grams: int) -> Dict[str, Any]:
        """Find best matching nutrition data"""
        # Clean food class name for matching
        clean_food_class = food_class.lower().strip().replace('_', ' ')
        
        # Filter data for this food class
        food_rows = self.nutrition_df[
            self.nutrition_df['label'].str.lower().str.replace('_', ' ') == clean_food_class
        ]
        
        if food_rows.empty:
            # Try partial matching
            food_rows = self.nutrition_df[
                self.nutrition_df['label'].str.lower().str.contains(clean_food_class, na=False)
            ]
        
        if food_rows.empty:
            return None
        
        # Find the closest weight match
        food_rows = food_rows.copy()
        food_rows['weight_diff'] = abs(food_rows['weight'] - serving_size_grams)
        closest_match = food_rows.loc[food_rows['weight_diff'].idxmin()]
        
        return closest_match.to_dict()
    
    def _format_nutrition_response(self, food_class: str, serving_size_grams: int, food_data: Dict) -> Dict[str, Any]:
        """Format nutrition response based on actual serving size"""
        base_weight = food_data['weight']
        multiplier = serving_size_grams / base_weight
        
        # Calculate nutrition values for requested serving size
        calories = round(food_data['calories'] * multiplier, 1)
        protein = round(food_data['protein'] * multiplier, 1)
        carbs = round(food_data['carbohydrates'] * multiplier, 1)
        fats = round(food_data['fats'] * multiplier, 1)
        
        # Optional nutrients (with defaults)
        fiber = round(food_data.get('fiber', 0) * multiplier, 1)
        sugars = round(food_data.get('sugars', 0) * multiplier, 1)
        sodium = round(food_data.get('sodium', 0) * multiplier, 1)
        
        return {
            'food_item': food_class,
            'serving_size_grams': serving_size_grams,
            'calories': calories,
            'macronutrients': {
                'protein_g': protein,
                'fat_g': fats,
                'carbs_g': carbs,
                'fiber_g': fiber,
                'sugars_g': sugars
            },
            'micronutrients': {
                'sodium_mg': sodium,
                'calcium_mg': 0.0,  # Not in your CSV
                'iron_mg': 0.0,     # Not in your CSV
                'vitamin_c_mg': 0.0  # Not in your CSV
            },
            'calorie_breakdown': {
                'from_protein': round(protein * 4, 1),
                'from_fat': round(fats * 9, 1),
                'from_carbs': round(carbs * 4, 1),
                'total_calculated': round((protein * 4) + (fats * 9) + (carbs * 4), 1)
            },
            'data_source': 'database'
        }
    
    def _get_estimated_nutrition(self, food_class: str, serving_size_grams: int) -> Dict[str, Any]:
        """Provide estimated nutrition when no data is found"""
        # Basic estimates per 100g (you can improve these)
        estimates_per_100g = {
            'calories': 200,
            'protein': 8,
            'carbs': 25,
            'fat': 8,
            'fiber': 3,
            'sugars': 5,
            'sodium': 300
        }
        
        multiplier = serving_size_grams / 100.0
        
        return {
            'food_item': food_class,
            'serving_size_grams': serving_size_grams,
            'calories': round(estimates_per_100g['calories'] * multiplier, 1),
            'macronutrients': {
                'protein_g': round(estimates_per_100g['protein'] * multiplier, 1),
                'fat_g': round(estimates_per_100g['fat'] * multiplier, 1),
                'carbs_g': round(estimates_per_100g['carbs'] * multiplier, 1),
                'fiber_g': round(estimates_per_100g['fiber'] * multiplier, 1),
                'sugars_g': round(estimates_per_100g['sugars'] * multiplier, 1)
            },
            'micronutrients': {
                'sodium_mg': round(estimates_per_100g['sodium'] * multiplier, 1),
                'calcium_mg': 0.0,
                'iron_mg': 0.0,
                'vitamin_c_mg': 0.0
            },
            'calorie_breakdown': {
                'from_protein': round(estimates_per_100g['protein'] * 4 * multiplier, 1),
                'from_fat': round(estimates_per_100g['fat'] * 9 * multiplier, 1),
                'from_carbs': round(estimates_per_100g['carbs'] * 4 * multiplier, 1),
                'total_calculated': round(((estimates_per_100g['protein'] * 4) + 
                                         (estimates_per_100g['fat'] * 9) + 
                                         (estimates_per_100g['carbs'] * 4)) * multiplier, 1)
            },
            'data_source': 'estimated'
        }
    
    async def get_available_foods(self) -> List[str]:
        """Get list of available food classes"""
        if self.nutrition_df is not None:
            return self.nutrition_df['label'].unique().tolist()
        return []
    
    async def get_database_size(self) -> int:
        """Get number of records in nutrition database"""
        if self.nutrition_df is not None:
            return len(self.nutrition_df)
        return 0
    
    async def add_food_data(self, food_data: Dict[str, Any]):
        """Add new food data to the database"""
        if self.nutrition_df is not None:
            new_row = pd.DataFrame([food_data])
            self.nutrition_df = pd.concat([self.nutrition_df, new_row], ignore_index=True)
        
    async def get_food_weights(self, food_class: str) -> List[int]:
        """Get available serving sizes for a food class"""
        if self.nutrition_df is not None:
            clean_food_class = food_class.lower().strip().replace('_', ' ')
            food_rows = self.nutrition_df[
                self.nutrition_df['label'].str.lower().str.replace('_', ' ') == clean_food_class
            ]
            return food_rows['weight'].tolist() if not food_rows.empty else []
        return []
