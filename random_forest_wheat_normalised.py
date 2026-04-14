import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


# -------------------------------
# 1. Load and preprocess data
# -------------------------------

df = pd.read_csv("src/agriculture_dataset.csv")

# Save basic stats if you want
df.describe().to_csv("features_described.csv")

print("Original number of rows:", len(df))

# Keep only wheat
df = df[df["Crop_Type"].str.lower() == "wheat"].copy()

# Remove invalid / missing rows
df = df[df["Expected_Yield"] >= 0]
df = df.dropna()

print("Number of wheat rows after cleaning:", len(df))

# Log transform skewed features
df["Canopy_Coverage"] = np.log1p(df["Canopy_Coverage"])
df["Rainfall"] = np.log1p(df["Rainfall"])
df["Chlorophyll_Content"] = np.log1p(df["Chlorophyll_Content"])
df["Organic_Matter"] = np.log1p(df["Organic_Matter"])
df["Wind_Speed"] = np.log1p(df["Wind_Speed"])


# -------------------------------
# 2. Define target and features
# -------------------------------

y = df["Crop_Health_Label"].astype(int).values

features = [
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

X = df[features].astype(float).values
y = df["Crop_Health_Label"].astype(int).values

print("Overall label counts:", np.bincount(y))

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
)

print("Train counts:", np.bincount(y_train))
print("Val counts:", np.bincount(y_val))
print("Test counts:", np.bincount(y_test))
# -------------------------------
# 3. Train / Validation / Test split
# -------------------------------

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train class balance (y=1 rate): {np.mean(y_train):.4f}")

# -------------------------------
# 4. Standardize numeric features
# -------------------------------
# Random Forest does not strictly need this,
# but since you already wanted preprocessing, we keep it.
# Important: fit ONLY on training data.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# -------------------------------
# 5. Hyperparameter tuning
# -------------------------------

param_grid = {
    "n_estimators": [75, 100, 125],
    "max_depth": [8, 10, 12, 15],
    "min_samples_split": [2, 5, 8, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nRunning Randomized Search...")
search.fit(X_train, y_train)

print("\nBest parameters found:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1 score: {search.best_score_:.4f}")


# -------------------------------
# 6. Validation evaluation
# -------------------------------

best_rf = search.best_estimator_

val_pred = best_rf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred, zero_division=0)
val_recall = recall_score(y_val, val_pred, zero_division=0)
val_f1 = f1_score(y_val, val_pred, zero_division=0)

print("\n--- Validation Results ---")
print("accuracy  :", round(val_accuracy, 4))
print("precision :", round(val_precision, 4))
print("recall    :", round(val_recall, 4))
print("f1 score  :", round(val_f1, 4))


# -------------------------------
# 7. Test evaluation
# -------------------------------

test_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
cm = confusion_matrix(y_test, test_pred)

TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

print("\n--- Wheat Random Forest Test Results ---")
print("n_samples :", len(X))
print("accuracy  :", round(accuracy, 4))
print("precision :", round(precision, 4))
print("recall    :", round(recall, 4))
print("f1 score  :", round(f1, 4))
print("confusion matrix:")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")

print("\nPrediction distribution:")
print("Validation:", np.bincount(val_pred))
print("Test:", np.bincount(test_pred))

print("\nActual distribution:")
print("Validation:", np.bincount(y_val))
print("Test:", np.bincount(y_test))

# -------------------------------
# 8. Feature importance
# -------------------------------

importances = best_rf.feature_importances_
print("\nFeature importances (ranked):")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")


# -------------------------------
# 9. Save model + scaler
# -------------------------------

with open("wheat_rf_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

with open("wheat_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved:")
print("  wheat_rf_model.pkl")
print("  wheat_scaler.pkl")