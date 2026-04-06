# CropDiseasePredictor
MAIS 202 Project
📌 Overview

This project is a machine learning web application designed to support agricultural decision-making. It includes two main tools:

Wheat Crop Health Predictor
Predicts whether a wheat crop is healthy or unhealthy based on environmental and agricultural data.
Crop Recommendation System (AI Advisor)
Suggests the best crop to plant based on climate and soil conditions such as temperature, rainfall, humidity, and soil characteristics.

Together, these tools help users both evaluate crop health and decide what to plant, making the system more practical and realistic.

Features
🌱 Predict wheat crop health (Healthy / Unhealthy)
🌍 Recommend best crop based on environmental conditions
🤖 Two AI systems:
Classification (health prediction)
Recommendation (crop selection)
📊 Multiple ML models:
Random Forest
Gradient Boosting
⚖️ Handles class imbalance
🔍 Confidence scores for predictions
🌐 Flask-based interactive web app
🧠 Machine Learning Systems
1. Wheat Crop Health Predictor
Binary classification problem
Input features:
Rainfall
Temperature
Soil moisture
Vegetation indices (NDVI, SAVI)
Chlorophyll content
etc.
Output:
Healthy / Unhealthy crop
2. Crop Recommendation System
Multi-class classification problem
Predicts the best crop to plant
Input features:
Temperature
Humidity
Rainfall
Soil pH
Nutrient levels (if used)
Output:
Recommended crop (e.g. wheat, rice, maize, etc.)

👉 This acts like an AI farming advisor

🖥️ Web Application

The web app allows users to access the first tool.

Pages
/ → Homepage
/try → input the different informations and then outputs the prediction

📂 Project Structure
project/
│
├── app.py
├── predict.py
│
├── wheat_rf_model.pkl
├── wheat_scaler.pkl
│
├── templates/
│   ├── layout.html
│   ├── index.html
│   ├── try.html
│
│
└── requirements.txt

⚙️ Installation
pip install -r requirements.txt
flask run

Then open:

http://127.0.0.1:5000/

📦 Requirements
flask
pandas
numpy
scikit-learn

🔬 Future Improvements
Integrate real-time weather APIs
Add map-based visualization
Use deep learning models
Deploy online (Render / Railway)
Add batch predictions

🧠 Key Insights
Crop health depends on multiple interacting environmental factors
Class imbalance significantly affects model performance
Feature transformations improve learning
AI can assist both diagnosis (health) and decision-making (what to plant)

👤 Author

Jean Eloi Lia Mohamed
2026