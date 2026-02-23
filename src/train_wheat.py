import pandas as pd
import numpy as np

# Load data + select Wheat 

df = pd.read_csv("data/agriculture_dataset.csv")
#print(df.head())
#print(df.columns)

df = df[df["Crop_Type"].str.lower() == "wheat"].copy()

y = df["Crop_Health_Label"].astype(int).values

features = [ 
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Soil_Moisture",
    "Soil_pH",
    "Organic_Matter",
    "Elevation_Data"
]

X = df[features].astype(float).values

# Standardize features

mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1.0
X = (X - mu) / sigma

# Train / Test Split

np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X)) 

train_idx = idx[:split]
test_idx = idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Regression 

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1-eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

n_features = X_train.shape[1]
w = np.zeros(n_features, dtype=float)
b = 0.0

lr = 0.05
epochs = 2000

for epoch in range(epochs): 
    z = X_train @ w + b
    y_prob = sigmoid(z)

    #gradients
    dw = (X_train.T @ (y_prob - y_train)) / len(y_train)
    db = float(np.mean(y_prob - y_train))

    w -= lr * dw
    b -= lr *db 

# Evaluate the results 
test_prob = sigmoid(X_test @ w + b)
test_pred = (test_prob >= 0.7).astype(int)

accuracy = float(np.mean(test_pred == y_test))

# Precision / Recall 
TP = int(np.sum((test_pred == 1) & (y_test == 1)))
FP = int(np.sum((test_pred == 1) & (y_test == 0)))
TN = int(np.sum((test_pred == 0) & (y_test == 0)))
FN = int(np.sum((test_pred == 0) & (y_test == 1)))

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0

print("\n--- Wheat results  ---")
print("n_samples:", len(X))
print("accuracy :", accuracy)
print("precision:", precision)
print("recall   :", recall)
print("confusion matrix:")
print(f"TP={TP}  FP={FP}")
print(f"FN={FN}  TN={TN}")

print(np.mean(test_prob))
print(np.min(test_prob), np.max(test_prob))