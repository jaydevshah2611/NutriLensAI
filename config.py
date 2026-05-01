"""
Configuration settings for Calorie AI project
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'food-101', 'food-101', 'images')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
STATIC_PATH = os.path.join(BASE_DIR, 'static')
UPLOAD_PATH = os.path.join(STATIC_PATH, 'uploads')

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Model settings - Optimized for CPU training
IMG_SIZE = 224
BATCH_SIZE = 64  # Increased for faster processing
EPOCHS = 2       # Quick training for usable model
LEARNING_RATE = 0.001
NUM_CLASSES = 101

# Training settings
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Calorie estimation settings
CALORIE_DATABASE_FILE = os.path.join(BASE_DIR, 'calorie_database.json')
INGREDIENT_DETECTION_THRESHOLD = 0.3

# Flask settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Food classes (will be loaded from dataset)
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]
