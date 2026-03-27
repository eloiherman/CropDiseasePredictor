import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('code/drafts/agriculture_dataset.csv')

df.describe().to_csv('code/drafts/features_described.csv')

print(len(df))

df = df[df['Expected_Yield'] >= 0] # remove rows with negative expected yield
df = df.dropna() # remove rows with missing values, does not remove any rows tho

### NORMALIZING SOME FEATURES ###

# log transform skewed features to reduce skewness and make them more normally distributed
df['Canopy_Coverage'] = np.log1p(df['Canopy_Coverage'])
df['Rainfall'] = np.log1p(df['Rainfall'])
#df['Expected_Yield'] = np.log1p(df['Expected_Yield'])
df['Chlorophyll_Content'] = np.log1p(df['Chlorophyll_Content'])
df['Organic_Matter'] = np.log1p(df['Organic_Matter'])
df['Wind_Speed'] = np.log1p(df['Wind_Speed'])

# features to normalize
num_features = [
    'Spatial_Resolution','Elevation_Data','Canopy_Coverage','NDVI','SAVI',
    'Chlorophyll_Content','Leaf_Area_Index','Temperature','Humidity',
    'Rainfall','Wind_Speed','Soil_Moisture','Soil_pH','Organic_Matter',
    'Water_Flow','Expected_Yield'
]

print(df[num_features].skew()) 

scaler = StandardScaler()

df[num_features] = scaler.fit_transform(df[num_features]) # transforms to mean 0 std dev 1

### IMPLEMENTING BINNING FOR SOME FEATURES ###

# need to test with model performance to see if binning is helpful or not


