import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# Make sure this path matches your project structure
CSV_PATH = "data/agriculture_dataset.csv"

df = pd.read_csv(CSV_PATH)

# Optional: filter to wheat only
wheat_only = True
if wheat_only:
    df = df[df["Crop_Type"].str.lower() == "wheat"].copy()

print("Data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# Split by health label
healthy = df[df["Crop_Health_Label"] == 1]
unhealthy = df[df["Crop_Health_Label"] == 0]

# Features to plot
features = [
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Soil_Moisture",
    "Soil_pH",
    "Organic_Matter",
    "Elevation_Data",
]

# 1) Histograms: healthy vs unhealthy for each feature
for feature in features:
    plt.figure(figsize=(8, 5))
    plt.hist(healthy[feature], bins=40, alpha=0.5, label="Healthy")
    plt.hist(unhealthy[feature], bins=40, alpha=0.5, label="Unhealthy")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"{feature} Distribution by Health Label")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 2) Scatter plots: feature vs health label
for feature in features:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[feature], df["Crop_Health_Label"], alpha=0.15)
    plt.xlabel(feature)
    plt.ylabel("Crop_Health_Label")
    plt.title(f"{feature} vs Crop Health Label")
    plt.tight_layout()
    plt.show()

# 3) Correlation matrix for numeric columns
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
