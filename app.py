from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html", prediction=None, patient=None)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
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


if __name__ == "__main__":
    app.run(debug=True)
