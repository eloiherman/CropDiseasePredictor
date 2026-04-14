import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data + select Wheat 

df = pd.read_csv("src/agriculture_dataset.csv")
df = df[df["Crop_Type"].str.lower() == "wheat"].copy()

y = df["Crop_Health_Label"].astype(int).values

# All numeric/usable features (dropped: GPS_Coordinates, Bounding_Boxes,
# Ground_Truth_Segmentation, Field_Boundaries — these are metadata strings)
features = [
    "Spatial_Resolution",
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
    "Pest_Hotspots",
    "Weed_Coverage",
    "Pest_Damage",
    "Crop_Growth_Stage",
    "Expected_Yield",
    "Water_Flow",
    "Drainage_Features",
]

X = df[features].astype(float).values

print(f"Using {len(features)} features on {len(X)} wheat samples")

# Train / Validation / Test Split (80/10/10) 
# No standardization needed for Random Forest

np.random.seed(42)
idx = np.random.permutation(len(X))
split1 = int(0.8 * len(X))
split2 = int(0.9 * len(X))

train_idx = idx[:split1]
val_idx   = idx[split1:split2]
test_idx  = idx[split2:]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Class balance (y=1 rate): {np.mean(y_train):.3f}")

# Hyperparameter Tuning via Randomized Search 

param_grid = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "max_features":      ["sqrt", "log2"],
    "class_weight":      ["balanced", None],
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nRunning Randomized Search (this may take a few minutes)...")
search.fit(X_train, y_train)

print("\nBest parameters found:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1 score: {search.best_score_:.4f}")

# Evaluate on validation set 

best_rf  = search.best_estimator_
val_pred = best_rf.predict(X_val)
val_f1   = f1_score(y_val, val_pred)
print(f"\nValidation F1: {val_f1:.4f}")

# Evaluate on test set 

test_pred = best_rf.predict(X_test)

accuracy  = accuracy_score(y_test,  test_pred)
precision = precision_score(y_test, test_pred)
recall    = recall_score(y_test,    test_pred)
f1        = f1_score(y_test,        test_pred)
cm        = confusion_matrix(y_test, test_pred)

TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

print("\n--- Wheat Random Forest Results ---")
print("n_samples :", len(X))
print("accuracy  :", round(accuracy,  4))
print("precision :", round(precision, 4))
print("recall    :", round(recall,    4))
print("f1 score  :", round(f1,        4))
print("confusion matrix:")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")

# Feature Importance 

importances = best_rf.feature_importances_
print("\nFeature importances (ranked):")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")

# Save model 

import pickle
with open("wheat_rf_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)
print("\nModel saved: wheat_rf_model.pkl")