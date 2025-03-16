import os
import pickle
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request

DiseasePrediction_blueprint = Blueprint("disease_prediction", __name__)

model_path = r"C:\Users\ADMIN\source\repos\MedicalPrediction\models\trained_model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

@DiseasePrediction_blueprint.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            glucose = float(request.form["glucose"])
            insulin = float(request.form["insulin"])
            blood_pressure = float(request.form["blood_pressure"])

            
            input_data = np.array([[age, bmi, glucose, insulin, blood_pressure]])

            
            prediction = model.predict(input_data)[0]

            
            result = "At Risk" if prediction == 1 else "Not at Risk"
            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", result=f"Error: {str(e)}")

    return render_template("index.html", result=None)
