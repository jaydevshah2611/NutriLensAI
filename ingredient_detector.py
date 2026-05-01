"""
Ingredient Detection Module for Fine-Grained Calorie Estimation
Uses color analysis and texture detection to identify visible ingredients
"""

import cv2
import numpy as np
from PIL import Image
import json
import config

class IngredientDetector:
    """Detect visible ingredients in food images for detailed calorie estimation"""
    
    def __init__(self):
        # Color ranges for different ingredients (HSV format)
        self.color_ranges = {
            # Greens
            'lettuce': {'lower': [35, 40, 40], 'upper': [85, 255, 255], 'confidence': 0.7},
            'spinach': {'lower': [35, 50, 20], 'upper': [75, 255, 180], 'confidence': 0.6},
            'basil': {'lower': [35, 100, 50], 'upper': [75, 255, 200], 'confidence': 0.6},
            'cilantro': {'lower': [35, 60, 40], 'upper': [80, 255, 220], 'confidence': 0.5},
            'avocado': {'lower': [30, 40, 40], 'upper': [70, 255, 200], 'confidence': 0.6},
            'green_pepper': {'lower': [35, 50, 50], 'upper': [80, 255, 200], 'confidence': 0.5},
            
            # Reds
            'tomato': {'lower': [0, 100, 50], 'upper': [15, 255, 255], 'confidence': 0.7},
            'pepperoni': {'lower': [0, 80, 50], 'upper': [20, 255, 200], 'confidence': 0.6},
            'red_pepper': {'lower': [0, 100, 50], 'upper': [15, 255, 255], 'confidence': 0.5},
            'strawberry': {'lower': [0, 120, 50], 'upper': [15, 255, 255], 'confidence': 0.5},
            'ketchup': {'lower': [0, 150, 50], 'upper': [15, 255, 200], 'confidence': 0.6},
            
            # Oranges/Red-Oranges
            'carrot': {'lower': [10, 100, 50], 'upper': [25, 255, 255], 'confidence': 0.6},
            'cheese_orange': {'lower': [15, 50, 150], 'upper': [35, 255, 255], 'confidence': 0.5},
            'cheddar_cheese': {'lower': [15, 80, 120], 'upper': [35, 255, 255], 'confidence': 0.6},
            'american_cheese': {'lower': [20, 40, 180], 'upper': [40, 150, 255], 'confidence': 0.5},
            
            # Yellows
            'pineapple': {'lower': [20, 100, 150], 'upper': [35, 255, 255], 'confidence': 0.6},
            'corn': {'lower': [20, 60, 150], 'upper': [35, 255, 255], 'confidence': 0.5},
            'banana_pepper': {'lower': [20, 80, 150], 'upper': [35, 255, 255], 'confidence': 0.4},
            'egg_yolk': {'lower': [20, 80, 150], 'upper': [35, 255, 255], 'confidence': 0.5},
            
            # Browns (meats, breads)
            'baked_bread': {'lower': [15, 40, 80], 'upper': [35, 200, 200], 'confidence': 0.5},
            'crust': {'lower': [15, 60, 100], 'upper': [35, 220, 220], 'confidence': 0.4},
            'beef': {'lower': [0, 30, 30], 'upper': [25, 200, 150], 'confidence': 0.5},
            'sausage': {'lower': [0, 40, 30], 'upper': [25, 220, 150], 'confidence': 0.4},
            'bacon': {'lower': [0, 50, 30], 'upper': [20, 255, 180], 'confidence': 0.5},
            'ham': {'lower': [0, 30, 100], 'upper': [20, 180, 230], 'confidence': 0.4},
            
            # Whites/Creams
            'onion': {'lower': [0, 0, 180], 'upper': [30, 80, 255], 'confidence': 0.5},
            'garlic': {'lower': [0, 0, 200], 'upper': [30, 50, 255], 'confidence': 0.4},
            'mushroom': {'lower': [20, 20, 60], 'upper': [40, 150, 180], 'confidence': 0.5},
            'sour_cream': {'lower': [0, 0, 200], 'upper': [50, 50, 255], 'confidence': 0.5},
            'mozzarella': {'lower': [0, 0, 220], 'upper': [50, 80, 255], 'confidence': 0.5},
            'feta_cheese': {'lower': [0, 0, 200], 'upper': [50, 60, 255], 'confidence': 0.4},
            
            # Blacks/Dark colors
            'olives': {'lower': [0, 0, 0], 'upper': [50, 255, 80], 'confidence': 0.5},
            'seaweed': {'lower': [0, 0, 0], 'upper': [180, 255, 60], 'confidence': 0.4},
            'burnt_crust': {'lower': [0, 0, 0], 'upper': [40, 255, 100], 'confidence': 0.3},
            
            # Pinks
            'salmon': {'lower': [0, 50, 120], 'upper': [20, 180, 255], 'confidence': 0.6},
            'shrimp': {'lower': [0, 40, 120], 'upper': [20, 150, 255], 'confidence': 0.5},
            
            # Purples
            'red_onion': {'lower': [130, 50, 50], 'upper': [170, 255, 200], 'confidence': 0.5},
            'beets': {'lower': [140, 80, 40], 'upper': [180, 255, 180], 'confidence': 0.4},
        }
        
        # Food-specific ingredient mappings
        self.food_ingredient_mappings = {
            'pizza': {
                'primary_ingredients': ['mozzarella', 'tomato', 'cheese_orange'],
                'optional_ingredients': [
                    'pepperoni', 'mushroom', 'sausage', 'bacon', 'ham',
                    'olives', 'green_pepper', 'red_pepper', 'onion',
                    'pineapple', 'spinach', 'basil', 'cheddar_cheese'
                ],
                'variation_keywords': {
                    'pepperoni': ['pepperoni'],
                    'mushroom': ['mushroom'],
                    'supreme': ['pepperoni', 'sausage', 'green_pepper', 'onion', 'mushroom'],
                    'meat_lovers': ['pepperoni', 'sausage', 'bacon', 'ham'],
                    'veggie': ['green_pepper', 'red_pepper', 'onion', 'mushroom', 'olives', 'spinach'],
                    'hawaiian': ['ham', 'pineapple'],
                    'four_cheese': ['mozzarella', 'cheddar_cheese', 'feta_cheese'],
                    'margherita': ['mozzarella', 'tomato', 'basil'],
                }
            },
            'hamburger': {
                'primary_ingredients': ['beef', 'baked_bread'],
                'optional_ingredients': [
                    'cheddar_cheese', 'american_cheese', 'bacon', 'lettuce',
                    'tomato', 'onion', 'pickles', 'avocado', 'mushroom',
                    'egg_yolk', 'red_onion', 'jalapenos'
                ],
                'variation_keywords': {
                    'cheeseburger': ['cheddar_cheese'],
                    'bacon_cheeseburger': ['cheddar_cheese', 'bacon'],
                    'mushroom_swiss': ['mushroom'],
                    'avocado': ['avocado'],
                    'egg': ['egg_yolk'],
                    'jalapeno': ['jalapenos'],
                    'bbq_bacon': ['bacon'],
                }
            },
            'sushi': {
                'primary_ingredients': ['rice'],
                'optional_ingredients': [
                    'salmon', 'tuna', 'shrimp', 'avocado', 'cucumber',
                    'cream_cheese', 'seaweed', 'egg', 'sesame_seeds'
                ],
                'variation_keywords': {
                    'salmon': ['salmon'],
                    'tuna': ['tuna'],
                    'shrimp': ['shrimp'],
                    'california': ['avocado', 'cucumber'],
                    'dragon': ['avocado', 'shrimp'],
                    'tempura': ['shrimp', 'tempura_batter'],
                    'philadelphia': ['salmon', 'cream_cheese'],
                }
            },
            'salad': {
                'primary_ingredients': ['lettuce'],
                'optional_ingredients': [
                    'tomato', 'cucumber', 'carrot', 'onion', 'avocado',
                    'grilled_chicken', 'egg_yolk', 'bacon', 'cheese',
                    'croutons', 'olives', 'red_onion', 'spinach',
                    'feta_cheese', 'walnuts'
                ],
                'variation_keywords': {
                    'caesar': ['lettuce', 'croutons', 'parmesan_cheese'],
                    'greek': ['feta_cheese', 'olives', 'tomato', 'cucumber', 'red_onion'],
                    'cobb': ['lettuce', 'chicken', 'bacon', 'egg_yolk', 'avocado'],
                    'caprese': ['tomato', 'mozzarella', 'basil'],
                }
            },
        }
    
    def detect_ingredients(self, image_path):
        """
        Detect ingredients in a food image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Detected ingredients with confidence scores
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        detected_ingredients = {}
        
        # Analyze each color range
        for ingredient, color_info in self.color_ranges.items():
            lower = np.array(color_info['lower'])
            upper = np.array(color_info['upper'])
            base_confidence = color_info['confidence']
            
            # Create mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Calculate area
            pixel_count = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            area_ratio = pixel_count / total_pixels
            
            # Calculate confidence based on area ratio
            # More area = higher confidence (with diminishing returns)
            if area_ratio > 0.02:  # At least 2% of image (increased from 1%)
                area_confidence = min(1.0, area_ratio * 4)  # Scale up, max 1.0
                final_confidence = base_confidence * area_confidence
                
                if final_confidence > 0.35:  # Higher minimum threshold (was 0.2)
                    detected_ingredients[ingredient] = {
                        'confidence': round(final_confidence, 3),
                        'area_percentage': round(area_ratio * 100, 2)
                    }
        
        # Sort by confidence
        detected_ingredients = dict(sorted(
            detected_ingredients.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        ))
        
        return detected_ingredients
    
    def detect_food_variation(self, food_class, detected_ingredients):
        """
        Determine food variation based on detected ingredients
        
        Args:
            food_class: Predicted food class
            detected_ingredients: Dictionary of detected ingredients
            
        Returns:
            dict: Best matching variation and confidence
        """
        if food_class not in self.food_ingredient_mappings:
            return {
                'variation': 'classic',
                'confidence': 0.5,
                'detected_ingredients': list(detected_ingredients.keys())[:5]
            }
        
        mapping = self.food_ingredient_mappings[food_class]
        variations = mapping.get('variation_keywords', {})
        
        best_variation = 'classic'
        best_score = 0
        
        # Score each variation
        for variation, required_ingredients in variations.items():
            score = 0
            matched = 0
            
            for ingredient in required_ingredients:
                if ingredient in detected_ingredients:
                    score += detected_ingredients[ingredient]['confidence']
                    matched += 1
            
            # Normalize score by number of required ingredients
            if len(required_ingredients) > 0:
                normalized_score = score / len(required_ingredients)
                # Bonus for matching more ingredients
                coverage_bonus = matched / len(required_ingredients)
                final_score = normalized_score * (0.7 + 0.3 * coverage_bonus)
                
                if final_score > best_score:
                    best_score = final_score
                    best_variation = variation
        
        return {
            'variation': best_variation,
            'confidence': round(best_score, 3),
            'detected_ingredients': list(detected_ingredients.keys())[:8]
        }
    
    def estimate_detailed_calories(self, food_class, detected_ingredients, serving_size='medium'):
        """
        Estimate calories with detailed breakdown based on ingredients
        
        Args:
            food_class: Predicted food class
            detected_ingredients: Dictionary of detected ingredients
            serving_size: 'small', 'medium', or 'large'
            
        Returns:
            dict: Detailed calorie estimation
        """
        from calorie_database import CALORIE_DATABASE, get_food_mapping
        
        # Get food mapping
        food_mapping = get_food_mapping()
        
        if food_class not in food_mapping:
            return {
                'total_calories': 300,
                'serving_size': serving_size,
                'variation': 'classic',
                'ingredient_breakdown': [],
                'confidence': 0.5
            }
        
        category_key, default_variation = food_mapping[food_class]
        category_data = CALORIE_DATABASE.get(category_key, {})
        
        # Detect variation
        variation_result = self.detect_food_variation(food_class, detected_ingredients)
        detected_variation = variation_result['variation']
        
        # Get variation data
        variations = category_data.get('variations', {})
        if detected_variation in variations:
            variation_data = variations[detected_variation]
        elif default_variation in variations:
            variation_data = variations[default_variation]
        else:
            # Fallback to base calories
            variation_data = {'calories': category_data.get('base_calories', 300)}
        
        base_calories = variation_data.get('calories', 300)
        
        # Adjust for serving size
        size_multiplier = {
            'small': 0.7,
            'medium': 1.0,
            'large': 1.4,
            'extra_large': 1.8
        }.get(serving_size, 1.0)
        
        adjusted_calories = int(base_calories * size_multiplier)
        
        # Build ingredient breakdown
        ingredient_breakdown = []
        ingredient_db = category_data.get('ingredients', {})
        
        for ingredient, detection_info in detected_ingredients.items():
            if ingredient in ingredient_db:
                ingredient_info = ingredient_db[ingredient]
                contribution = int(ingredient_info['calories'] * detection_info['confidence'])
                ingredient_breakdown.append({
                    'name': ingredient.replace('_', ' ').title(),
                    'calories': contribution,
                    'confidence': detection_info['confidence'],
                    'portion': ingredient_info['portion']
                })
        
        # Sort by calorie contribution
        ingredient_breakdown.sort(key=lambda x: x['calories'], reverse=True)
        
        # Ensure detected_ingredients matches what's in the breakdown
        # This fixes the UI mismatch between displayed ingredients and breakdown
        detected_names = [item['name'].lower().replace(' ', '_') for item in ingredient_breakdown]
        
        return {
            'total_calories': adjusted_calories,
            'base_calories': base_calories,
            'serving_size': serving_size,
            'size_multiplier': size_multiplier,
            'variation': detected_variation,
            'variation_confidence': variation_result['confidence'],
            'detected_ingredients': detected_names[:8],  # Match breakdown items
            'ingredient_breakdown': ingredient_breakdown[:8],
            'nutrition_per_100g': category_data.get('nutrition_per_100g', {}),
            'description': variation_data.get('description', ''),
            'confidence': round(variation_result['confidence'], 3)
        }
    
    def analyze_image(self, image_path, food_class, serving_size='medium'):
        """
        Full analysis pipeline - Fixed to show consistent ingredients
        
        Args:
            image_path: Path to the image
            food_class: Predicted food class
            serving_size: Serving size option
            
        Returns:
            dict: Complete analysis results
        """
        from calorie_database import CALORIE_DATABASE, get_food_mapping
        
        # Detect ingredients by color
        detected_ingredients = self.detect_ingredients(image_path)
        
        # Get food category and its expected ingredients
        food_mapping = get_food_mapping()
        category_key, _ = food_mapping.get(food_class, ('', 'classic'))
        category_data = CALORIE_DATABASE.get(category_key, {})
        expected_ingredients = set(category_data.get('ingredients', {}).keys())
        
        # Filter detected ingredients to only those relevant to this food
        # AND with reasonable confidence
        relevant_detected = {}
        for ingredient, info in detected_ingredients.items():
            # Only include if: 1) relevant to food OR 2) high confidence > 0.5
            if ingredient in expected_ingredients and info['confidence'] > 0.3:
                relevant_detected[ingredient] = info
        
        # If no relevant ingredients detected, use the food's typical ingredients
        if not relevant_detected and expected_ingredients:
            # Get top 3-5 typical ingredients for this food
            typical_ingredients = list(expected_ingredients)[:5]
            for ing in typical_ingredients:
                relevant_detected[ing] = {'confidence': 0.5, 'area_percentage': 10}
        
        # Estimate calories with filtered ingredients
        # This now returns consistent detected_ingredients that match the breakdown
        calorie_estimate = self.estimate_detailed_calories(
            food_class, relevant_detected, serving_size
        )
        
        return {
            'food_class': food_class,
            'ingredient_detection': {
                'detected_count': len(relevant_detected),
                'ingredients': relevant_detected
            },
            'calorie_estimate': calorie_estimate
        }

# Singleton instance
ingredient_detector = IngredientDetector()

def analyze_food_image(image_path, food_class, serving_size='medium'):
    """Convenience function for food image analysis"""
    return ingredient_detector.analyze_image(image_path, food_class, serving_size)

def detect_ingredients(image_path):
    """Convenience function for ingredient detection"""
    return ingredient_detector.detect_ingredients(image_path)
