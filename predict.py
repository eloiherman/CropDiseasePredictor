import pickle
import numpy as np

with open("wheat_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("wheat_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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

    prediction = "Healthy wheat crop" if pred == 1 else "Unhealthy wheat crop"
    confidence = round(float(np.max(probs)) * 100, 2)

    return prediction, confidence