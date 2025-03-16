import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Folders
DATASET_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#Saving the dataframe
def save_dataframe(df, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(DATASET_DIR, f"{os.path.splitext(filename)[0]}_{timestamp}.csv")
    df.to_csv(file_path, index=False)
    print(f"CSV saved: {file_path}")
    return file_path

# Loading the selected features
file_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\datasets\selected_features.csv"
df = pd.read_csv(file_path)

# Splitting the data
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

save_dataframe(pd.concat([X_train, y_train], axis=1), "train_data.csv")
save_dataframe(pd.concat([X_test, y_test], axis=1), "test_data.csv")

param_grid = {"n_estimators": [50, 100, 200]}
model = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
model.fit(X_train, y_train)

model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model.best_estimator_, file)
print(f"Model saved: {model_path}")
