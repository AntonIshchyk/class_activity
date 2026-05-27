from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html", prediction=None, patient=None)


@app.route("/predict", methods=["POST"])
def predict():
    age = request.form.get("age", type=float) or 0.0
    hypertension = request.form.get("hypertension", type=int) or 0
    heart_disease = request.form.get("heart_disease", type=int) or 0
    avg_glucose = request.form.get("avg_glucose", type=float) or 0.0
    bmi = request.form.get("bmi", type=float) or 0.0
    gender = request.form.get("male", type=str) or "female"
    smoking_status = request.form.get("smoking_status", type=str) if request.form.get("smoking_status", type=str) in ["never smoked", "formerly smoked", "smokes"] else "Unknown"

    payload = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose": avg_glucose,
        "bmi": bmi,
        "gender": gender,
        "smoking_status": smoking_status,
    }
    encode_features(payload)
    return (
        jsonify(
            {
                "message": "Predict endpoint is ready.",
                "input": payload,
                "prediction": None,
            }
        ),
        200,
    )
    
def encode_features(payload):
    payload['gender'] = 1 if payload['gender'] == 'male' else 0
    payload['smoking_status'] = {
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': -1
    }.get(payload['smoking_status'], -1)

if __name__ == "__main__":
    app.run(debug=True)
