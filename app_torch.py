"""
Calorie AI - Flask Backend API (PyTorch Version)
Handles image uploads, food classification, and calorie estimation
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import config
from ingredient_detector import analyze_food_image
from calorie_database import CALORIE_DATABASE, get_food_mapping

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None
class_indices = None
food_mapping = None
device = None

# Image transforms
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(num_classes):
    """Create ResNet50 model with custom classifier"""
    model = models.resnet50(weights=None)  # We'll load weights from checkpoint
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

def load_model():
    """Load the trained model and class indices - downloads from cloud if not available locally"""
    global model, class_indices, food_mapping, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    class_indices_path = os.path.join(config.MODEL_SAVE_PATH, 'class_indices.json')
    
    # Load class indices first to get number of classes
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print(f"Loaded {len(class_indices)} classes")
    else:
        print(f"Warning: Class indices not found at {class_indices_path}")
        return False
    
    # Check for model - try local first, then download from cloud
    final_model_path = model_path
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found locally at {model_path}")
        print(f"📥 Attempting to download from cloud storage...")
        
        try:
            # Try to download from cloud
            from model_downloader import get_model_path, get_model_path_serverless
            
            # Check if running in serverless environment (Vercel)
            if os.environ.get('VERCEL') or not os.path.exists(config.MODEL_SAVE_PATH):
                print("Using serverless download method...")
                final_model_path = get_model_path_serverless()
            else:
                print("Using standard download method...")
                final_model_path = get_model_path()
                
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("Please ensure model is available locally or MODEL_URL is set correctly")
            model = None
            
            # Load food mapping even if model fails
            food_mapping = get_food_mapping()
            print("Food mapping loaded")
            return False
    
    # Load model from the final path (local or downloaded)
    if os.path.exists(final_model_path):
        print(f"Loading model from {final_model_path}")
        
        num_classes = len(class_indices)
        model = create_model(num_classes)
        
        checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model not found at {final_model_path}")
        model = None
    
    # Load food mapping
    food_mapping = get_food_mapping()
    print("Food mapping loaded")
    
    return model is not None

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)

def predict_food(image_path):
    """Predict food class from image"""
    if model is None:
        return None, None
    
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top 5 predictions
    probs = probabilities.cpu().numpy()[0]
    top_5_indices = np.argsort(probs)[-5:][::-1]
    
    results = []
    for idx in top_5_indices:
        class_name = class_indices.get(str(idx), class_indices.get(int(idx), 'unknown'))
        confidence = float(probs[idx])
        results.append({
            'class': class_name,
            'confidence': round(confidence, 4)
        })
    
    return results[0]['class'], results

def get_food_info(food_class):
    """Get food information from database"""
    if food_mapping and food_class in food_mapping:
        category_key, variation_key = food_mapping[food_class]
        category_data = CALORIE_DATABASE.get(category_key, {})
        
        variations = category_data.get('variations', {})
        variation_data = variations.get(variation_key, {})
        
        return {
            'category': category_key,
            'base_calories': category_data.get('base_calories', 0),
            'per_serving': category_data.get('per_serving', ''),
            'nutrition_per_100g': category_data.get('nutrition_per_100g', {}),
            'variations': list(variations.keys()) if variations else [],
            'default_variation': variation_key,
            'default_description': variation_data.get('description', '')
        }
    
    return None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_indices) if class_indices else 0,
        'device': str(device) if device else 'none'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict food class from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get serving size from form data (default to medium)
    serving_size = request.form.get('serving_size', 'medium')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(config.UPLOAD_PATH, filename)
    file.save(filepath)
    
    try:
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        top_class, top_5_predictions = predict_food(filepath)
        
        if top_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get food info
        food_info = get_food_info(top_class)
        
        # Analyze ingredients and estimate calories
        analysis = analyze_food_image(filepath, top_class, serving_size)
        
        # Get variation details with calories for alternatives
        variation_details = {}
        if food_info and food_info.get('variations'):
            category_key = food_info.get('category', '')
            category_data = CALORIE_DATABASE.get(category_key, {})
            variations_data = category_data.get('variations', {})
            for var_name in food_info['variations']:
                if var_name in variations_data:
                    variation_details[var_name] = {
                        'calories': variations_data[var_name].get('calories', 0),
                        'description': variations_data[var_name].get('description', '')
                    }
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'food_class': top_class,
                'confidence': top_5_predictions[0]['confidence'],
                'top_5_predictions': top_5_predictions
            },
            'food_info': food_info,
            'variation_details': variation_details,
            'analysis': analysis,
            'image_url': f'/static/uploads/{filename}'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    
    finally:
        # Clean up uploaded file after some time (optional)
        # os.remove(filepath)
        pass

@app.route('/api/variations/<food_class>', methods=['GET'])
def get_variations(food_class):
    """Get available variations for a food class"""
    food_info = get_food_info(food_class)
    
    if food_info:
        return jsonify({
            'success': True,
            'food_class': food_class,
            'variations': food_info.get('variations', []),
            'default_variation': food_info.get('default_variation', 'classic')
        })
    
    return jsonify({
        'success': False,
        'error': 'Food class not found'
    }), 404

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all food classes"""
    if class_indices:
        classes = list(class_indices.values())
        return jsonify({
            'success': True,
            'count': len(classes),
            'classes': sorted(classes)
        })
    
    return jsonify({
        'success': False,
        'error': 'Classes not loaded'
    }), 500

@app.route('/api/calorie-database', methods=['GET'])
def get_calorie_database():
    """Get full calorie database"""
    return jsonify({
        'success': True,
        'database': CALORIE_DATABASE
    })

@app.route('/api/analyze-ingredients', methods=['POST'])
def analyze_ingredients():
    """
    Analyze calories based on voice/text input of ingredients
    Expected JSON: {"ingredients": ["rice", "chicken", "vegetables"], "serving_size": 1.0}
    """
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({
                'success': False,
                'error': 'No ingredients provided'
            }), 400
        
        ingredients = data.get('ingredients', [])
        serving_size = float(data.get('serving_size', 1.0))
        
        if not ingredients:
            return jsonify({
                'success': False,
                'error': 'Empty ingredients list'
            }), 400
        
        # Calculate calories based on ingredients
        total_calories = 0
        ingredient_breakdown = []
        
        # Common ingredient calorie database (per 100g or standard serving)
        ingredient_calories = {
            'rice': 130, 'white rice': 130, 'brown rice': 112,
            'chicken': 165, 'chicken breast': 165, 'grilled chicken': 165,
            'beef': 250, 'steak': 271, 'ground beef': 250,
            'fish': 206, 'salmon': 208, 'tuna': 132,
            'egg': 155, 'eggs': 155,
            'bread': 265, 'toast': 265,
            'pasta': 131, 'noodles': 138,
            'potato': 77, 'potatoes': 77,
            'tomato': 18, 'tomatoes': 18,
            'onion': 40, 'onions': 40,
            'garlic': 149,
            'carrot': 41, 'carrots': 41,
            'peas': 81,
            'corn': 86,
            'spinach': 23,
            'lettuce': 15,
            'cucumber': 15,
            'pepper': 20, 'bell pepper': 20,
            'broccoli': 34,
            'cauliflower': 25,
            'mushroom': 22, 'mushrooms': 22,
            'cheese': 402, 'cheddar': 402, 'mozzarella': 280,
            'milk': 42,
            'butter': 717,
            'oil': 884, 'olive oil': 884,
            'sugar': 387,
            'salt': 0,
            'flour': 364,
            'oats': 389,
            'yogurt': 59,
            'banana': 89, 'bananas': 89,
            'apple': 52, 'apples': 52,
            'orange': 47, 'oranges': 47,
            'grapes': 69,
            'strawberry': 32, 'strawberries': 32,
            'chocolate': 546,
            'nuts': 607, 'almonds': 579, 'walnuts': 654,
            'honey': 304,
            'jam': 278,
            'cream': 345,
            'mayonnaise': 680,
            'ketchup': 101,
            'mustard': 66,
            'soy sauce': 53,
            'vinegar': 21,
            'lemon': 29, 'lemon juice': 25,
            'lime': 30,
            'ginger': 80,
            'turmeric': 312,
            'cumin': 375,
            'coriander': 298,
            'basil': 22,
            'oregano': 265,
            'thyme': 101,
            'rosemary': 131,
            'parsley': 36,
            'cilantro': 23,
            'mint': 70,
            'green chili': 40, 'chili': 40,
            'black pepper': 251,
            'cinnamon': 247,
            'cardamom': 311,
            'cloves': 274,
            'bay leaf': 313,
            'coconut': 354, 'coconut milk': 230,
            'peanut': 567, 'peanuts': 567,
            'sesame': 573,
            'tofu': 76,
            'lentils': 116,
            'beans': 347, 'kidney beans': 347, 'black beans': 341,
            'chickpeas': 164,
            'pomegranate': 83,
            'mango': 60, 'mangoes': 60,
            'pineapple': 50,
            'watermelon': 30,
            'papaya': 43,
            'guava': 68,
            'litchi': 66,
            'jackfruit': 95,
            'custard apple': 94,
            'sugar cane': 58,
            'jaggery': 383,
            'paneer': 265,
            'ghee': 900,
            'dal': 116, 'lentil': 116,
            'sambar': 65,
            'rasam': 35,
            'curry': 120, 'vegetable curry': 90, 'chicken curry': 140,
            'roti': 264, 'chapati': 264,
            'naan': 310,
            'paratha': 290,
            'dosa': 168,
            'idli': 58,
            'vada': 290,
            'samosa': 262,
            'pakora': 250,
            'biryani': 180, 'chicken biryani': 200, 'vegetable biryani': 160,
            'pulao': 150,
            'fried rice': 163,
            'noodles': 138,
            'burger': 295,
            'pizza': 266, 'cheese pizza': 280,
            'sandwich': 250, 'club sandwich': 300,
            'wrap': 280,
            'salad': 33, 'green salad': 25, 'caesar salad': 180,
            'soup': 45, 'tomato soup': 40, 'chicken soup': 70,
            'dal makhani': 150,
            'butter chicken': 280,
            'palak paneer': 190,
            'chana masala': 140,
            'aloo gobi': 110,
            'bhindi masala': 90,
            'baingan bharta': 100,
            'rajma': 140,
            'chole': 150,
            'pav bhaji': 180,
            'vada pav': 290,
            'misal pav': 250,
            'dhokla': 160,
            'khandvi': 130,
            'thepla': 220,
            'khichdi': 120,
            'pongal': 180,
            'upma': 140,
            'poha': 150,
            'sabudana': 170,
            'chivda': 450,
            'bhel puri': 180,
            'sev puri': 200,
            'pani puri': 120,
            'dahi puri': 150,
            'aloo tikki': 180,
            'papdi chaat': 210,
            'gol gappa': 30,
            'jalebi': 460,
            'gulab jamun': 320,
            'rasgulla': 186,
            'kaju katli': 450,
            'barfi': 400,
            'ladoo': 380,
            'halwa': 350,
            'kheer': 150,
            'payasam': 160,
            'sheera': 320,
            'basundi': 250,
            'shrikhand': 220,
            'amrakhand': 240,
            'mattha': 85,
            'lassi': 110,
            'chaas': 42,
            'sharbat': 80,
            'rokad sharbat': 70,
            'kokum juice': 45,
            'solkadhi': 65,
            'aam panna': 90,
            'jaljeera': 35,
            'chhaachh': 40,
            'masala chaas': 50,
            'buttermilk': 42,
            'tak': 45,
            'neer more': 42,
            'curd rice': 140,
            'yogurt rice': 140,
            'dahi bhat': 140,
            'thayir sadam': 140,
            'mosaru anna': 140,
            'dahi chawal': 140,
            'golgappa': 30,
            'fulki': 30,
            'puchka': 30,
            'gupchup': 30,
            'tikki': 180,
            'aloo chaat': 160,
            'dahi bhalla': 150,
            'dahi vada': 150,
            'dahi bada': 150,
            'thayir vadai': 150,
            'mosaru vade': 150,
            'perugu garelu': 150,
            'dahi bhalle': 150,
            'raita': 80,
            'koshimbir': 70,
            'pachadi': 75,
            'palya': 90,
            'bhaji': 120,
            'sabzi': 100,
            'sukhi bhaji': 85,
            'bharit': 110,
            'zhunka': 180,
            'pitla': 170,
            'zunka bhakar': 200,
            'thaalipeeth': 220,
            'bharli vangi': 160,
            'stuffed brinjal': 160,
            'bharit': 110,
            'vangi bhath': 180,
            'masala bhaat': 170,
            'tamarind rice': 160,
            'lemon rice': 150,
            'coconut rice': 180,
            'tomato rice': 165,
            'curd rice': 140,
            'yogurt rice': 140,
            'sambar rice': 160,
            'rasam rice': 145,
            'bisibelebath': 200,
            'vangi bath': 180,
            'puliyogare': 170,
            'pulihora': 170,
            'tamarind rice': 160,
            'chitranna': 150,
            'lemon rice': 150,
            'nimbehannu chitranna': 150,
            'nimmakaya pulihora': 160,
            'kobbari annam': 180,
            'thengai sadam': 180,
            'coconut rice': 180,
            'ellu sadam': 190,
            'sesame rice': 190,
            'pudina rice': 140,
            'mint rice': 140,
            'coriander rice': 135,
            'cilantro rice': 135,
            'kothamalli sadam': 135,
            'methi rice': 145,
            'fenugreek rice': 145,
            'venthaya sadam': 145,
            'ulava charu': 120,
            'horsegram soup': 120,
            'pesarattu': 150,
            'moong dal dosa': 150,
            'uttapam': 160,
            'onion uttapam': 175,
            'tomato uttapam': 170,
            'masala dosa': 200,
            'mysore masala dosa': 220,
            'rava dosa': 140,
            'sooji dosa': 140,
            'semolina dosa': 140,
            'neer dosa': 120,
            'water dosa': 120,
            'set dosa': 160,
            'sponge dosa': 160,
            'appam': 150,
            'kallappam': 155,
            'vellayappam': 150,
            'palappam': 160,
            'noolappam': 145,
            'idiappam': 130,
            'string hoppers': 130,
            'stringhopper': 130,
            'sevai': 130,
            'santhakai': 135,
            'kozhukattai': 140,
            'modak': 220,
            'kadubu': 215,
            'sukhiyan': 180,
            'sugiyan': 180,
            'bonda': 190,
            'mysore bonda': 200,
            'batata vada': 210,
            'aloo bonda': 210,
            'mirchi bajji': 170,
            'chili bajji': 170,
            'milagai bajji': 170,
            'menasinakai bajji': 175,
            'capsicum bajji': 160,
            'brinjal bajji': 155,
            'badanekai bajji': 155,
            'vangyache bharit': 155,
            'keerai': 30,
            'greens': 30,
            'spinach': 23,
            'amaranth': 23,
            'bachali': 25,
            'ponnaganti': 25,
            'thotakura': 23,
            'agathi keerai': 20,
            'mulai keerai': 22,
            'palak': 23,
            'fenugreek': 49,
            'methi': 49,
            'mint': 70,
            'pudina': 70,
            'coriander': 23,
            'cilantro': 23,
            'kothamalli': 23,
            'kothambari': 23,
            'curry leaves': 108,
            'karivepaku': 108,
            'karibevu': 108,
            'murraya': 108,
            'drumstick leaves': 120,
            'moringa': 120,
            'murungai keerai': 120,
            'nugge soppu': 120,
            'sajna patta': 120,
            'beetroot': 43,
            'beet root': 43,
            'chukandar': 43,
            'radish': 16,
            'mooli': 16,
            'daikon': 18,
            'turnip': 28,
            'shalgam': 28,
            'sweet potato': 86,
            'shakarkandi': 86,
            'yam': 118,
            'suran': 118,
            'elephant foot': 118,
            'tapioca': 159,
            'cassava': 159,
            'maravalli kizhangu': 159,
            'kuchi kizhangu': 159,
            'colocasia': 142,
            'arbi': 142,
            'seppankizhangu': 142,
            'chamadumpa': 142,
            'raw banana': 89,
            'vazhakka': 89,
            'plantain': 122,
            'nendran': 122,
            'elephant yam': 118,
            'chena': 118,
            'cheena': 118,
            'oint': 40,
            'shallot': 72,
            'sambar onion': 72,
            'cherry tomato': 18,
            'grape tomato': 18,
            ' heirloom tomato': 18,
            'ridge gourd': 20,
            'beerakaya': 20,
            'turiya': 20,
            'sponge gourd': 20,
            'bottle gourd': 15,
            'sorakaya': 15,
            'lauki': 15,
            'dudhi': 15,
            'bitter gourd': 34,
            'kakarakaya': 34,
            'karle': 34,
            'karela': 34,
            'ash gourd': 13,
            'boodida gumadikaya': 13,
            'petha': 13,
            'winter melon': 13,
            'snake gourd': 19,
            'potlakaya': 19,
            'padavalanga': 19,
            'chichinda': 19,
            'ivy gourd': 30,
            'tindora': 30,
            'tendli': 30,
            'tondlekai': 30,
            'kovaikai': 30,
            'pointed gourd': 20,
            'parwal': 20,
            'potol': 20,
            'lotus stem': 74,
            'kamal kakdi': 74,
            'tamara thandu': 74,
            'bamboo shoot': 27,
            'karira': 27,
            'heart of palm': 25,
            'palmito': 25,
            'banana flower': 51,
            'vazhaipoo': 51,
            'arati puvvu': 51,
            'mocha': 51,
            'banana stem': 39,
            'vazhaithandu': 39,
            'arati dantu': 39,
            'thor': 39,
            'tender coconut': 68,
            'ilaneer': 68,
            'elaneer': 68,
            'coconut water': 19,
            'coconut cream': 330,
            'thick coconut milk': 230,
            'thin coconut milk': 150,
            'mustard': 66,
            'rai': 66,
            'kadugu': 66,
            'asafoetida': 40,
            'hing': 40,
            'perungayam': 40,
            'fenugreek seeds': 323,
            'methi dana': 323,
            'vendhayam': 323,
            'poppy seeds': 525,
            'khus khus': 525,
            'gasagase': 525,
            'posto': 525,
            'cashew': 553,
            'kaju': 553,
            'mundhiri': 553,
            'pista': 557,
            'pistachio': 557,
            'dates': 282,
            'khajoor': 282,
            'pericham palam': 282,
            'kharik': 282,
            'fig': 74,
            'anjeer': 74,
            'athi pazham': 74,
            'dumur': 74,
            'raisins': 299,
            'kishmish': 299,
            'drakshi': 299,
            'kismis': 299,
            'dried apricot': 241,
            'jardalu': 241,
            'walnut': 654,
            'akhrot': 654,
            'akharot': 654,
            'almond': 579,
            'badam': 579,
            'badami': 579,
            'hazelnut': 628,
            'pecan': 691,
            'brazil nut': 659,
            'macadamia': 718,
            'pinenut': 673,
            'chironji': 653,
            'charoli': 653,
            'piyal': 653,
            'nellekai': 45,
            'amla': 48,
            'amla candy': 320,
            'amla murabba': 310,
            'gooseberry': 44,
            'karonda': 55,
            'christ thorn': 55,
            'ber': 77,
            'bor': 77,
            'jujube': 79,
            'regi pallu': 79,
            'tamarind': 239,
            'imli': 239,
            'puli': 239,
            'hunise hannu': 239,
            'chintapandu': 239,
            'tetul': 239,
            'kokum': 60,
            'bhirind': 60,
            'amsool': 60,
            'fish': 206,
            'machli': 206,
            'meen': 206,
            'singhara': 206,
            'prawn': 99,
            'shrimp': 99,
            'jhinga': 99,
            'eraal': 99,
            'chingri': 99,
            'crab': 97,
            'kekd': 97,
            'nandu': 97,
            'kankda': 97,
            'lobster': 89,
            'squid': 92,
            'octopus': 82,
            'cuttlefish': 79,
            'mussels': 172,
            'oyster': 81,
            'clam': 148,
            'scallop': 111,
            'turmeric': 312,
            'haldi': 312,
            'manjal': 312,
            'arishina': 312,
            'red chili': 40,
            'lal mirch': 40,
            'milagai': 40,
            'menasinakai': 40,
            'green chili': 40,
            'hari mirch': 40,
            'pachai milagai': 40,
            'kashmiri chili': 40,
            'byadgi chili': 40,
            'guntur chili': 40,
            'black pepper': 251,
            'kali mirch': 251,
            'milagu': 251,
            'menasu': 251,
            'gol morich': 251,
            'cumin': 375,
            'jeera': 375,
            'seeragam': 375,
            'jira': 375,
            'coriander seed': 298,
            'dhaniya': 298,
            'malli': 298,
            'dhane': 298,
            'fennel': 345,
            'saunf': 345,
            'sombu': 345,
            'mouri': 345,
            'star anise': 337,
            'cinnamon': 247,
            'dalchini': 247,
            'pattai': 247,
            'daruchini': 247,
            'clove': 274,
            'laung': 274,
            'lavangam': 274,
            'lobongo': 274,
            'cardamom': 311,
            'elaichi': 311,
            'elakkai': 311,
            'elach': 311,
            'bay leaf': 313,
            'tej patta': 313,
            'brinji elai': 313,
            'tej pata': 313,
            'nutmeg': 525,
            'jaiphal': 525,
            'jathikai': 525,
            'jaayiful': 525,
            'mace': 475,
            'javitri': 475,
            'jathipoo': 475,
            'jaayitri': 475,
            'saffron': 310,
            'kesar': 310,
            'kumkumapoo': 310,
            'zaafran': 310,
            'rose water': 0,
            'gulab jal': 0,
            'panneer': 0,
            'rosha jal': 0,
            'kewra': 0,
            'pandan': 0,
            'rampe': 0,
            'kewda': 0,
            'orange blossom': 0,
            'neroli': 0,
            'edible gold': 0,
            'varak': 0,
            'silver leaf': 0,
            'edible silver': 0,
            'rose petal': 0,
            'gulkand': 200,
            'rose preserve': 200,
            'paan': 10,
            'beetle leaf': 10,
            'vethalai': 10,
            'paan leaf': 10,
            'supari': 400,
            'betel nut': 400,
            'adakka': 400,
            'copra': 354,
            'dried coconut': 354,
            'sukha nariyal': 354,
            'sugar': 387,
            'shakkar': 387,
            'cheeni': 387,
            'panchdara': 387,
            'brown sugar': 380,
            'gur': 383,
            'jaggery': 383,
            'vellam': 383,
            'gul': 383,
            'palm sugar': 380,
            'palm jaggery': 380,
            'taad gur': 380,
            'karupatti': 380,
            'honey': 304,
            'shahad': 304,
            'then': 304,
            'mou': 304,
            'maple syrup': 260,
            'date syrup': 270,
            'molasses': 290,
            'golden syrup': 310,
            'corn syrup': 300,
            'agave': 310,
            'stevia': 0,
            'artificial sweetener': 0,
            'sugarfree': 0,
            'saccharin': 0,
            'aspartame': 0,
            'sucralose': 0,
            'wheat': 327,
            'gehu': 327,
            'godhumai': 327,
            'gom': 327,
            'atta': 340,
            'whole wheat flour': 340,
            'maida': 364,
            'all purpose flour': 364,
            'refined flour': 364,
            'maida mavu': 364,
            'besan': 387,
            'gram flour': 387,
            'kadalai mavu': 387,
            'sattu': 380,
            'barley flour': 345,
            'jowar': 349,
            'sorghum': 349,
            'cholam': 349,
            'bajra': 361,
            'pearl millet': 361,
            'kambu': 361,
            'sajje': 361,
            'ragi': 336,
            'finger millet': 336,
            'kezhvaragu': 336,
            'nachni': 336,
            'millets': 378,
            'foxtail millet': 473,
            'kangni': 473,
            'thinai': 473,
            'navane': 473,
            'kakum': 473,
            'little millet': 341,
            'kutki': 341,
            'saamai': 341,
            'same': 341,
            'kodo millet': 353,
            'kodon': 353,
            'varagu': 353,
            'haraka': 353,
            'arikelu': 353,
            'barnyard millet': 300,
            'sanwa': 300,
            'kuthiraivali': 300,
            'oodalu': 300,
            'proso millet': 356,
            'panivaragu': 356,
            'cheena': 356,
            'baragu': 356,
            'amaranth': 371,
            'rajgira': 371,
            'ramdana': 371,
            'quinoa': 368,
            'oats': 389,
            'jai': 389,
            'yavalakki': 389,
            'semolina': 360,
            'suji': 360,
            'rava': 360,
            'rawa': 360,
            'sooji': 360,
            'vermicelli': 350,
            'sevai': 350,
            'seviyan': 350,
            'shavige': 350,
            'semiya': 350,
            'santhakai': 350,
            'puffed rice': 402,
            'murmura': 402,
            'pori': 402,
            'kurmura': 402,
            'muri': 402,
            'puffed wheat': 380,
            'rolled oats': 379,
            'instant oats': 362,
            'steel cut oats': 375,
            'rice flakes': 346,
            'aval': 346,
            'poha': 346,
            'atukulu': 346,
            'chiura': 346,
            'flattened rice': 346,
            'beaten rice': 346,
            'cornflakes': 357,
            'muesli': 340,
            'granola': 450,
            'wheat germ': 382,
            'rice bran': 375,
            'germinated grain': 145,
            'sprouted moong': 30,
            'sprouted matki': 30,
            'sprouted methi': 40,
            'sprouted chana': 45,
            'sprouted wheat': 150,
            'soy chunk': 336,
            'meal maker': 336,
            'nutrela': 336,
            'soy granules': 336,
            'textured vegetable protein': 262,
            'seitan': 370,
            'mock meat': 200,
            'plant based meat': 250,
            'tofu': 76,
            'bean curd': 76,
            'silk tofu': 55,
            'firm tofu': 144,
            'tempeh': 193,
            'edamame': 122,
            'seaweed': 43,
            'nori': 35,
            'kelp': 43,
            'wakame': 45,
            'kombu': 43,
            'spirulina': 290,
            'chlorella': 325,
            'nutritional yeast': 325,
            'brewer yeast': 325,
            'miso': 199,
            'soybean paste': 199,
            'doenjang': 200,
            'natto': 212,
            'fermented soybean': 212,
            'kimchi': 32,
            'sauerkraut': 19,
            'pickle': 17,
            'achaar': 17,
            'uragai': 17,
            'loncha': 17,
            'bharwa': 25,
            'avakaya': 20,
            'mango pickle': 20,
            'lime pickle': 18,
            'lemon pickle': 18,
            'nimbu ka achaar': 18,
            'chili pickle': 15,
            'garlic pickle': 40,
            'ginger pickle': 35,
            'carrot pickle': 25,
            'cauliflower pickle': 22,
            'turnip pickle': 20,
            'radish pickle': 18,
            'onion pickle': 22,
            'green chili pickle': 15,
            'mixed pickle': 30,
            'sweet pickle': 120,
            'chutney': 85,
            'pachadi': 85,
            'thuvaiyal': 90,
            'chatni': 85,
            'coconut chutney': 140,
            'tomato chutney': 70,
            'onion chutney': 80,
            'ginger chutney': 60,
            'mint chutney': 50,
            'coriander chutney': 55,
            'garlic chutney': 110,
            'dry coconut chutney': 480,
            'gun powder': 450,
            'milagai podi': 450,
            'karam podi': 450,
            'podi': 450,
            'chutney powder': 450,
            'curry powder': 325,
            'sambar powder': 310,
            'rasam powder': 300,
            'biryani masala': 280,
            'garam masala': 300,
            'pav bhaji masala': 320,
            'chaat masala': 340,
            'kitchen king masala': 310,
            'meat masala': 290,
            'chicken masala': 285,
            'fish masala': 275,
            'egg curry masala': 280,
            'paneer masala': 270,
            'tandoori masala': 260,
            'tikka masala': 270,
            'kadai masala': 290,
            'kolhapuri masala': 310,
            'malvani masala': 305,
            'goda masala': 295,
            'kala masala': 300,
            'bafat masala': 285,
            'saambar masala': 310,
            'rasam masala': 300,
            'puliyodarai mix': 280,
            'puliogare mix': 280,
            'bisi bele bath masala': 295,
            'vangi bath masala': 285,
            'chitranna mix': 270,
            'bisibelebath mix': 295,
            'pudina chutney powder': 310,
            'karivepaku podi': 310,
            'curry leaf powder': 310,
            'flaxseed': 534,
            'alsi': 534,
            'agase': 534,
            'tisi': 534,
            'chia seed': 486,
            'basil seed': 22,
            'sabja': 22,
            'falooda seed': 22,
            'sunflower seed': 584,
            'pumpkin seed': 559,
            'watermelon seed': 557,
            'muskmelon seed': 557,
            'cucumber seed': 557,
            'ash gourd seed': 557,
            'melon seed': 557,
            'char magaz': 557,
            'musk melon seed': 557,
            'lotus seed': 89,
            'makhana': 89,
            'fox nut': 89,
            'phool makhana': 89,
            'thamarai vithai': 89,
            'kamal gatta': 89,
            'water lily seed': 89,
            'prickly pear': 41,
            'cactus fruit': 41,
            'sabra': 41,
            'falsa': 61,
            'phalsa': 61,
            'currant': 56,
            'kishmish': 56,
            'cranberry': 46,
            'blueberry': 57,
            'raspberry': 52,
            'blackberry': 43,
            'mulberry': 43,
            'sea buckthorn': 82,
            'kiwi': 61,
            'avocado': 160,
            'butter fruit': 160,
            'makkhan phal': 160,
            'dragon fruit': 60,
            'pitaya': 60,
            'passion fruit': 97,
            'rambutan': 82,
            'lychee': 66,
            'longan': 60,
            'mangosteen': 73,
            'durian': 147,
            'jackfruit': 95,
            'breadfruit': 103,
            'breadnut': 217,
            'champak': 95,
            'rose apple': 25,
            'plum': 46,
            'aloo bukhara': 46,
            'jamun': 62,
            'java plum': 62,
            'black plum': 62,
            'nagapazham': 62,
            'nerale hannu': 62,
            'jambul': 62,
            'kala khatta': 62,
            'kokum': 60,
            'bhirind': 60,
            'garcinia': 60,
            'kokam': 60,
            'ratamba': 60,
            'aman shundi': 60,
            'bilimbi': 36,
            'cucumber tree': 36,
            'bilimbi fruit': 36,
            'kamias': 36,
            'star fruit': 31,
            'carambola': 31,
            'kamrakh': 31,
            'karmal': 31,
            'thamaraithai': 31,
            'chikoo': 83,
            'sapodilla': 83,
            'sapota': 83,
            'chiku': 83,
            'ciku': 83,
            'sapathi': 83,
            'chikku': 83,
            'tamarind': 239,
            'imli': 239,
            'puli': 239,
            'hunise hannu': 239,
            'chintapandu': 239,
            'tetul': 239,
            'sampalok': 239,
            'amchur': 258,
            'mango powder': 258,
            'dried mango': 319,
            'dried fig': 249,
            'anjeer': 249,
            'dried apricot': 241,
            'dried peach': 239,
            'dried pear': 262,
            'dried plum': 240,
            'prune': 240,
            'dried date': 282,
            'dried currant': 283,
            'dried berry': 325,
            'dried cherry': 325,
            'dried apple': 243,
            'apple rings': 243,
            'dried banana': 346,
            'banana chips': 519,
            'jackfruit chips': 432,
            'tapioca chips': 545,
            'potato chips': 536,
            'plantain chips': 520,
            'yam chips': 480,
            'sweet potato chips': 520,
            'taro chips': 480,
            'vegetable chips': 460,
            'okra chips': 340,
            'bitter gourd chips': 310,
            'carrot chips': 410,
            'beetroot chips': 380,
            'kale chips': 450,
            'spinach chips': 480,
            'seaweed chips': 390,
            'nori chips': 340,
            'rice chips': 480,
            'soy chips': 470,
            'lentil chips': 380,
            'chickpea chips': 390,
            'multigrain chips': 450,
            'pita chips': 456,
            'tortilla chips': 489,
            'nachos': 489,
            'corn chips': 497,
            'potato stick': 520,
            'cheese puff': 530,
            'onion ring': 440,
            'french fries': 312,
            'finger chips': 312,
            'potato wedge': 290,
            'curly fries': 320,
            'waffle fries': 310,
            'steak fries': 300,
            'crinkle cut': 315,
            'shoestring fries': 325,
            'hash brown': 265,
            'rosti': 265,
            'potato pancake': 265,
            'latke': 265,
            'boxty': 265,
            'tater tot': 255,
            'potato croquette': 280,
            'aloo tikki': 280,
            'potato cutlet': 280,
            'batata vada': 280,
            'bond': 280,
            'potato bond': 280,
            'batata chop': 280,
            'aloo chop': 280,
            'potato mash': 113,
            'aloo bharta': 113,
            'chokha': 113,
            'mashed potato': 113,
            'potato puree': 113,
            'aloo puree': 113,
            'potato soup': 85,
            'potato stew': 105,
            'potato curry': 125,
            'aloo curry': 125,
            'potato bhaji': 115,
            'aloo bhaji': 115,
            'potato fry': 195,
            'aloo fry': 195,
            'potato roast': 185,
            'aloo roast': 185,
            'aloo sukha': 185,
            'potato dry': 185,
            'potato poriyal': 185,
            'aloo poriyal': 185,
            'potato mezhukkupuratti': 185,
            'aloo thoran': 185,
            'potato thoran': 185,
            'aloo upperi': 185,
            'potato upperi': 185,
            'aloo ki sabzi': 115,
            'potato sabzi': 115,
            'aloo matar': 110,
            'potato peas': 110,
            'aloo gobi': 105,
            'potato cauliflower': 105,
            'aloo palak': 95,
            'potato spinach': 95,
            'aloo methi': 100,
            'potato fenugreek': 100,
            'aloo shimla mirch': 100,
            'potato capsicum': 100,
            'aloo baingan': 110,
            'potato brinjal': 110,
            'aloo tamatar': 105,
            'potato tomato': 105,
            'aloo pyaaz': 110,
            'potato onion': 110,
            'aloo pyaz': 110,
            'dum aloo': 180,
            'baked potato': 161,
            'jacket potato': 161,
            'roasted potato': 161,
            'grilled potato': 161,
            'barbecue potato': 161,
            'potato salad': 135,
            'german potato salad': 145,
            'potato gratin': 190,
            'au gratin potato': 190,
            'scalloped potato': 180,
            'potato dauphinois': 180,
            'potato casserole': 175,
            'potato pie': 220,
            'shepherd pie': 220,
            'cottage pie': 220,
            'potato puff': 210,
            'potato souffle': 210,
            'potato kugel': 195,
            'potato knish': 185,
            'potato bread': 265,
            'potato roll': 275,
            'potato bun': 280,
            'potato flatbread': 245,
            'potato naan': 285,
            'potato paratha': 290,
            'aloo paratha': 290,
            'potato kulcha': 280,
            'aloo kulcha': 280,
            'potato stuffed bread': 265,
            'potato roti': 245,
            'aloo roti': 245,
            'potato thepla': 235,
            'aloo thepla': 235,
            'potato poori': 320,
            'aloo poori': 320,
            'potato kachori': 380,
            'aloo kachori': 380,
            'potato samosa': 250,
            'aloo samosa': 250,
            'potato pastry': 280,
            'potato puff pastry': 310,
            'potato patty': 270,
            'potato cake': 240,
            'latke': 240,
            'potato fritter': 240,
            'potato pancake': 240,
            'potato waffle': 260,
            'potato doughnut': 380,
            'spudnut': 380,
            'potato dessert': 210,
            'potato pudding': 210,
            'potato halwa': 240,
            'sweet potato halwa': 280,
            'potato sweet': 220,
            'potato candy': 380,
            'potato chocolate': 420,
            'potato beverage': 85,
            'potato juice': 85,
            'potato milk': 95,
            'potato coffee': 45,
            'potato tea': 35
        }
        
        # Process each ingredient
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            # Find matching calorie value
            calories_per_100g = 0
            matched_name = ingredient_lower
            
            # Direct match
            if ingredient_lower in ingredient_calories:
                calories_per_100g = ingredient_calories[ingredient_lower]
            else:
                # Try to find partial match
                for known_ingredient, cal in ingredient_calories.items():
                    if ingredient_lower in known_ingredient or known_ingredient in ingredient_lower:
                        calories_per_100g = cal
                        matched_name = known_ingredient
                        break
            
            # Estimate amount (assume 100g if not specified)
            estimated_grams = 100
            
            # Check for quantity hints in the ingredient name
            import re
            quantity_match = re.search(r'(\d+)\s*(g|gram|kg|ml|cup|tbsp|tsp)', ingredient_lower)
            if quantity_match:
                amount = float(quantity_match.group(1))
                unit = quantity_match.group(2)
                if unit in ['kg', 'kilo']:
                    estimated_grams = amount * 1000
                elif unit in ['cup']:
                    estimated_grams = amount * 240  # Approximate
                elif unit in ['tbsp', 'tablespoon']:
                    estimated_grams = amount * 15
                elif unit in ['tsp', 'teaspoon']:
                    estimated_grams = amount * 5
                else:
                    estimated_grams = amount
            
            # Calculate calories for this ingredient
            ingredient_cals = (calories_per_100g * estimated_grams / 100) * serving_size
            total_calories += ingredient_cals
            
            ingredient_breakdown.append({
                'name': ingredient.title(),
                'matched_to': matched_name.title() if matched_name != ingredient_lower else None,
                'grams': estimated_grams * serving_size,
                'calories_per_100g': calories_per_100g,
                'calories': round(ingredient_cals, 1),
                'percentage': 0  # Will calculate after total
            })
        
        # Calculate percentages
        if total_calories > 0:
            for item in ingredient_breakdown:
                item['percentage'] = round((item['calories'] / total_calories) * 100, 1)
        
        # Sort by calorie contribution
        ingredient_breakdown.sort(key=lambda x: x['calories'], reverse=True)
        
        return jsonify({
            'success': True,
            'input_ingredients': ingredients,
            'total_calories': round(total_calories, 1),
            'serving_size': serving_size,
            'breakdown': ingredient_breakdown,
            'food_name': 'Custom Mix (' + ', '.join(ingredients[:3]) + ('...' if len(ingredients) > 3 else '') + ')',
            'confidence': 'Voice Input'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Load model on module import (for Vercel serverless)
model_loaded = load_model()

if __name__ == '__main__':
    if not model_loaded:
        print("\n" + "!" * 60)
        print("WARNING: Model not loaded!")
        print("Please train the model using: python train_model_torch.py")
        print("!" * 60 + "\n")
    
    # Run Flask app locally
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

# Vercel serverless handler
# This allows the app to work with Vercel's serverless functions
try:
    from vercel_wsgi import make_lambda_handler
    handler = make_lambda_handler(app)
except ImportError:
    # If vercel_wsgi is not installed, use Flask's built-in handling
    pass
