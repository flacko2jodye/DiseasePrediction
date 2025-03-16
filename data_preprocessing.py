import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def save_dataframe(df, filename="processed_data.csv"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(DATASET_DIR, f"{os.path.splitext(filename)[0]}_{timestamp}.csv")
        df.to_csv(file_path, index=False)
        print(f"CSV saved successfully: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        return None

file_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\DiseasePrediction\datasets\diabetes.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.fillna(df.mean(), inplace=True)

processed_file = save_dataframe(df, "cleaned_data.csv")

# Exploratory Data Analysis
print("Summary Statistics:")
print(df.describe())

sns.pairplot(df, diag_kind="kde")
plt.savefig(os.path.join(DATASET_DIR, "pairplot.png"))
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.savefig(os.path.join(DATASET_DIR, "correlation_matrix.png"))
plt.show()
