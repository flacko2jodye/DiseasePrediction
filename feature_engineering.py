import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from datetime import datetime

DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def save_dataframe(df, filename="feature_selected_data.csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(DATASET_DIR, f"{os.path.splitext(filename)[0]}_{timestamp}.csv")
    df.to_csv(file_path, index=False)
    print(f"Feature-selected CSV saved: {file_path}")
    return file_path

file_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\datasets\cleaned_data_20250317_040211.csv"
df = pd.read_csv(file_path)

# One-hot encoding for categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

if "Outcome" in df.columns:
    df["Outcome"] = df["Outcome"].astype(int)  

# Feature selection using Mutual Information
X = df.drop(columns=["Outcome"])  
y = df["Outcome"]

selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected["Outcome"] = y.values  

selected_features_path = os.path.join(DATASET_DIR, "selected_features.csv")
df_selected.to_csv(selected_features_path, index=False)

print(f"Selected features saved successfully: {selected_features_path}")