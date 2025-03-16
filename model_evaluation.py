import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

model_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\models\trained_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

test_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\datasets\test_data_20250317_040333.csv"
df_test = pd.read_csv(test_path)
X_test = df_test.drop(columns=["Outcome"])
y_test = df_test["Outcome"]

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(" Classification Report:\n", report)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(RESULTS_DIR, f"classification_report_{timestamp}.txt")
with open(report_path, "w") as file:
    file.write(report)

print(f"Report saved: {report_path}")
