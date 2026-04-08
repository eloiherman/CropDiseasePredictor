import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FEATURE_ORDER = [
    "Elevation_Data",
    "Canopy_Coverage",
    "NDVI",
    "SAVI",
    "Chlorophyll_Content",
    "Leaf_Area_Index",
    "Crop_Stress_Indicator",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Soil_Moisture",
    "Soil_pH",
    "Organic_Matter",
    "Weed_Coverage",
    "Pest_Damage",
    "Water_Flow",
]

with (BASE_DIR / "wheat_rf_model.pkl").open("rb") as f:
    model = pickle.load(f)

with (BASE_DIR / "wheat_scaler.pkl").open("rb") as f:
    scaler = pickle.load(f)

if hasattr(model, "n_jobs"):
    model.n_jobs = 1


def get_feature_importance_map():
    if not hasattr(model, "feature_importances_"):
        return {}
    return {
        feature_name: float(importance)
        for feature_name, importance in zip(MODEL_FEATURE_ORDER, model.feature_importances_)
    }


def get_model_class_count():
    classes = getattr(model, "classes_", None)
    return int(len(classes)) if classes is not None else 0


def make_prediction(form):
    features = [
        float(form["Elevation_Data"]),
        float(form["Canopy_Coverage"]),
        float(form["NDVI"]),
        float(form["SAVI"]),
        float(form["Chlorophyll_Content"]),
        float(form["Leaf_Area_Index"]),
        float(form["Temperature"]),
        float(form["Humidity"]),
        float(form["Rainfall"]),
        float(form["Wind_Speed"]),
        float(form["Soil_Moisture"]),
        float(form["Soil_pH"]),
        float(form["Organic_Matter"]),
        float(form["Water_Flow"]),
        float(form["Weed_Coverage"]),
        float(form["Crop_Stress_Indicator"]),
        float(form["Pest_Damage"])
    ]

    features[2] = np.log1p(features[1])
    features[9] = np.log1p(features[8])
    features[5] = np.log1p(features[4])
    features[13] = np.log1p(features[12])
    features[10] = np.log1p(features[9])

    X = np.array([features], dtype=float)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    class_probability_map = {
        int(class_label): round(float(probability) * 100, 2)
        for class_label, probability in zip(model.classes_, probs)
    }
    healthy_probability = class_probability_map.get(1, 0.0)
    unhealthy_probability = class_probability_map.get(0, 0.0)
    prediction = "Healthy wheat crop" if pred == 1 else "Unhealthy wheat crop"
    confidence = max(healthy_probability, unhealthy_probability)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "healthy": healthy_probability,
            "unhealthy": unhealthy_probability,
        },
    }
