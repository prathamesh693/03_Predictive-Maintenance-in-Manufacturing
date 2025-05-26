import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

# Load dataset
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/02_Data/predictive_maintenance.csv")

# Inspect dataset
print("Initial shape:", df.shape)
print(df.head())

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Handle missing values if any (example: fill with median)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns if any
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Separate features and target
target_col = 'Failure Type'  
X = df.drop(target_col, axis=1)
y = df[target_col]

# Remove outliers using IQR method for numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
    X = X[mask]
    y = y[mask]

print("Shape after outlier removal:", X.shape)

# Normalize numeric columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Feature selection: select top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Create DataFrame with selected features
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected[target_col] = y.values

# PCA for dimensionality reduction to 5 components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_selected.drop(target_col, axis=1))
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
df_pca[target_col] = df_selected[target_col].values

# Save final preprocessed data
df_pca.to_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/02_Data/Preprocessed_data.csv", index=False)
print("Final preprocessed data saved.")
