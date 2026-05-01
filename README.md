<div align="center">

# 🥗 NutriLens AI
### *Smart Food Recognition & Calorie Analysis*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-black.svg)](https://flask.palletsprojects.com)
[![Vercel](https://img.shields.io/badge/Vercel-Ready-000000.svg)](https://vercel.com)

**AI-powered web app that identifies food from images, detects ingredients, and estimates calories with detailed nutrition breakdown.**

<img src="https://img.shields.io/badge/Voice%20Input-🎙️-purple.svg" alt="Voice Input">
<img src="https://img.shields.io/badge/Ingredient%20Editor-✏️-green.svg" alt="Ingredient Editor">
<img src="https://img.shields.io/badge/Cloud%20Model-☁️-blue.svg" alt="Cloud Model">

[🌐 Live Demo](https://your-app.vercel.app) • [📖 Docs](DEPLOY.md) • [🚀 Deploy Guide](VERCEL_DEPLOY_CHECKLIST.md)

</div>

---

## ✨ Features

### 🎯 Core Features
| Feature | Description |
|---------|-------------|
| **📸 Image Upload** | Drag & drop food photos for instant analysis |
| **🧠 AI Prediction** | 101 food categories with 85-90% accuracy |
| **🥗 Ingredient Detection** | Automatic detection of visible ingredients |
| **📊 Calorie Breakdown** | Ingredient-level calorie contribution |
| **🍽️ Serving Sizes** | Small, Medium, Large, Extra-Large options |

### 🚀 Advanced Features
| Feature | Description |
|---------|-------------|
| **🎙️ Voice Input** | Speak ingredients naturally - AI understands and calculates |
| **✏️ Ingredient Editor** | Add/remove ingredients after analysis & recalculate calories |
| **☁️ Cloud Model** | Model auto-downloads from Google Drive (no Git LFS needed) |
| **🎨 Beautiful UI** | Modern glassmorphism design with lavender-peach-mint theme |
| **📱 Responsive** | Works on desktop, tablet, and mobile |

---

## 🎬 Demo

### Image Analysis
```
📸 Upload Image → 🧠 AI Analyzes → 🥗 Detects Ingredients → 📊 Shows Calories
```

### Voice Input
```
🎙️ "I had rice, chicken curry, and dal" → 📊 Instant calorie breakdown
```

### Ingredient Editor
```
🥗 AI Detects: Rice, Chicken
✏️ You Add: Ghee, Spices
🔄 Recalculate → Updated calorie count
```

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Flask API     │────▶│   PyTorch Model │
│  (HTML/CSS/JS)  │     │   (app_torch.py)│     │  (ResNet50)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Voice Input    │     │ Ingredient      │     │ Calorie         │
│  (Web Speech API)│     │ Detection       │     │ Database        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Backend** | Flask, Flask-CORS |
| **AI/ML** | PyTorch, TorchVision, ResNet50 |
| **Image Processing** | OpenCV, PIL |
| **Deployment** | Vercel / Railway / Render |
| **Cloud Storage** | Google Drive / Hugging Face |

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/nutri-lens-ai.git
cd nutri-lens-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Locally
```bash
# The model will auto-download from Google Drive on first run
python app_torch.py

# Open browser
curl http://localhost:5000
```

### 3️⃣ Deploy to Vercel
```bash
# Push to GitHub (no model file - it downloads automatically!)
git add .
git commit -m "Deploy with cloud model"
git push origin main

# Deploy
vercel --prod
```

**[📖 Detailed Deployment Guide →](VERCEL_DEPLOY_CHECKLIST.md)**

---

## 📂 Project Structure

```
nutri-lens-ai/
├── 🐍 app_torch.py              # Main Flask application
├── 📥 model_downloader.py       # Downloads model from cloud
├── 🧠 calorie_database.py       # 1000+ ingredient calorie data
├── 👁️ ingredient_detector.py   # Image ingredient detection
├── ⚙️ config.py                 # Configuration settings
├── 📋 requirements.txt          # Python dependencies
├── 🚀 vercel.json              # Vercel deployment config
├── 📝 Procfile                 # Railway/Render config
├── 🎨 templates/
│   └── index.html              # Beautiful UI frontend
├── 📁 static/uploads/          # User uploads
└── 🤖 models/
    └── .gitkeep                # Empty - model downloads from cloud!
```

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/predict` | POST | Upload image & get analysis |
| `/api/analyze-ingredients` | POST | Analyze ingredients from voice/text |
| `/api/calorie-database` | GET | Full calorie database |
| `/api/health` | GET | Health check |

### Example: Analyze Image
```bash
curl -X POST -F "image=@pizza.jpg" -F "serving_size=medium" \
  http://localhost:5000/api/predict
```

### Example: Voice Input
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"ingredients":["rice","chicken","dal"],"serving_size":1.0}' \
  http://localhost:5000/api/analyze-ingredients
```

---

## 🌈 Features in Detail

### 🎙️ Voice Input
- Uses Web Speech API (browser built-in)
- Natural language understanding
- 1000+ ingredient recognition
- Real-time calorie calculation

### ✏️ Ingredient Editor
- Add missing ingredients
- Remove incorrect detections
- Click "Recalculate" for instant update
- Suggested ingredients for quick add

### ☁️ Cloud Model Hosting
- Model stored on Google Drive / Hugging Face
- Auto-downloads on first run
- No Git LFS needed
- Repository stays lightweight (~50KB)

---

## 📊 Supported Foods

**101 Food Categories** including:
- 🍕 Pizza (14 variations)
- 🍔 Hamburger (14 variations)
- 🍣 Sushi, 🍜 Ramen, 🥗 Salad
- 🥘 Biryani, 🍛 Curry, 🥙 Wraps
- 🥞 Pancakes, 🍰 Cake, 🍩 Donuts
- And 90+ more!

**1000+ Ingredients** in database including:
- Vegetables: tomato, onion, spinach, etc.
- Proteins: chicken, beef, fish, tofu, etc.
- Carbs: rice, bread, pasta, noodles, etc.
- Spices: turmeric, cumin, garam masala, etc.
- Dairy: cheese, milk, butter, ghee, etc.

---

## 🚀 Deployment Options

| Platform | Best For | Setup Difficulty |
|----------|----------|------------------|
| **Vercel** | Quick deploy, serverless | ⭐ Easy |
| **Railway** | Persistent storage | ⭐ Easy |
| **Render** | Full-stack apps | ⭐⭐ Medium |
| **Hugging Face Spaces** | ML demos | ⭐ Easy |

**[☁️ External Model Hosting Guide →](EXTERNAL_MODEL_DEPLOY.md)**

---

## 🏆 Performance

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 85-90% (Top-1) |
| **Top-5 Accuracy** | ~97% |
| **Inference Time** | 100-200ms (CPU) |
| **Ingredient Detection** | 50-100ms |
| **Model Size** | ~104 MB |
| **App Size (GitHub)** | ~50 KB |

---

## 🛡️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_URL` | Yes | Google Drive / Hugging Face download link |
| `VERCEL` | Auto | Set by Vercel (detects serverless mode) |

---

## 📝 License

This project is for **educational purposes**.

---

## 🙏 Credits

- **Dataset**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) by ETH Zurich
- **Model**: ResNet50 (PyTorch torchvision)
- **UI Design**: Glassmorphism with lavender-peach-mint theme
- **Icons**: Emoji native support

---

<div align="center">

**Made with ❤️ and 🥗**

[🌐 Live Demo](https://your-app.vercel.app) • [⭐ Star this repo](https://github.com/YOUR_USERNAME/nutri-lens-ai) • [🐛 Report Issue](../../issues)

</div>
