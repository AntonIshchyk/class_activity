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
    smoking_status = (
        request.form.get("smoking_status", type=str)
        if request.form.get("smoking_status", type=str)
        in ["never smoked", "formerly smoked", "smokes"]
        else "Unknown"
    )

    # Store original values for patient dict
    raw_values = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose": avg_glucose,
        "bmi": bmi,
        "gender": gender,
        "smoking_status": smoking_status,
    }

    payload = raw_values.copy()
    input_array = encode_features(payload)

    # 2.5.4 Perform prediction
    probability = model.predict_proba(input_array)[0, 1]

    # 2.5.5 Map the probability to a risk level
    if probability < 0.25:
        risk_level = "LOW"
        color = "success"
        message = "No immediate escalation. Continue standard protocol."
    elif probability < 0.55:
        risk_level = "MODERATE"
        color = "warning"
        message = "Recommend neurological assessment within 2 hours."
    else:
        risk_level = "HIGH"
        color = "danger"
        message = "Urgent — refer for immediate neurological evaluation."

    # 2.5.6 Build prediction and patient dictionaries
    prediction = {
        "risk_level": risk_level,
        "risk_color": color,
        "risk_pct": round(probability * 100),
        "risk_msg": message,
    }

    patient = {
        "age": raw_values["age"],
        "hypertension": "Yes" if raw_values["hypertension"] == 1 else "No",
        "heart_disease": "Yes" if raw_values["heart_disease"] == 1 else "No",
        "avg_glucose": raw_values["avg_glucose"],
        "bmi": raw_values["bmi"],
        "gender": raw_values["gender"],
        "smoking_status": raw_values["smoking_status"],
    }

    return render_template("index.html", prediction=prediction, patient=patient)


def encode_features(payload):
    payload["gender"] = 1 if payload["gender"] == "male" else 0
    payload["smoking_status"] = {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": -1,
    }.get(payload["smoking_status"], -1)

    # 2.5.3 Build the input array
    input_array = np.array(
        [
            [
                payload["age"],
                payload["hypertension"],
                payload["heart_disease"],
                payload["avg_glucose"],
                payload["bmi"],
                payload["gender"],
                payload["smoking_status"],
            ]
        ]
    )
    return input_array


def load_model():
    global model
    model = joblib.load("stroke_model.pkl")


if __name__ == "__main__":
    load_model()
    app.run(debug=True)
