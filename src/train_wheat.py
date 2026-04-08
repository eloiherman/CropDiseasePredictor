from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "agriculture_dataset.csv"

FEATURES = [
    "Elevation_Data",
    "Canopy_Coverage",
    "NDVI",
    "SAVI",
    "Chlorophyll_Content",
    "Leaf_Area_Index",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Soil_Moisture",
    "Soil_pH",
    "Organic_Matter",
    "Water_Flow",
    "Weed_Coverage",
    "Crop_Stress_Indicator",
    "Pest_Damage",
]

LOG_TRANSFORM_FEATURES = [
    "Canopy_Coverage",
    "Chlorophyll_Content",
    "Rainfall",
    "Wind_Speed",
    "Organic_Matter",
]


def load_wheat_data():
    df = pd.read_csv(DATASET_PATH)
    df = df[df["Crop_Type"].str.lower() == "wheat"].copy()
    df = df[df["Expected_Yield"] >= 0].dropna()

    X = df[FEATURES].astype(float).copy()
    for feature in LOG_TRANSFORM_FEATURES:
        X[feature] = np.log1p(X[feature].clip(lower=0))

    y = df["Crop_Health_Label"].astype(int).to_numpy()
    return X, y


def choose_threshold(y_true, probabilities):
    best_score = None
    best_snapshot = None

    for threshold in np.linspace(0.30, 0.70, 81):
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        # Favor the metrics requested by the project update:
        # F1 first, then recall, then accuracy.
        score = (0.50 * f1) + (0.30 * recall) + (0.20 * accuracy)

        if best_score is None or score > best_score:
            best_score = score
            best_snapshot = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best_snapshot


def print_metrics(title, y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- {title} ---")
    print("accuracy :", accuracy)
    print("precision:", precision)
    print("recall   :", recall)
    print("F1 score :", f1)
    print("confusion matrix:")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")


def main():
    X, y = load_wheat_data()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=4000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    val_probabilities = model.predict_proba(X_val_scaled)[:, 1]
    best_threshold = choose_threshold(y_val, val_probabilities)

    val_predictions = (val_probabilities >= best_threshold["threshold"]).astype(int)
    test_probabilities = model.predict_proba(X_test_scaled)[:, 1]
    test_predictions = (test_probabilities >= best_threshold["threshold"]).astype(int)

    print("\n--- Wheat Training Setup ---")
    print("n_samples :", len(X))
    print("n_features:", len(FEATURES))
    print("train size:", len(X_train))
    print("val size  :", len(X_val))
    print("test size :", len(X_test))
    print("positive class share:", float(np.mean(y)))
    print("selected threshold  :", round(best_threshold["threshold"], 3))

    print_metrics("Validation Results", y_val, val_predictions)
    print_metrics("Test Results", y_test, test_predictions)


if __name__ == "__main__":
    main()
