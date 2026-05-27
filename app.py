from flask import Flask, jsonify, request


app = Flask(__name__)


@app.get("/")
def home():
	return jsonify({"message": "Home endpoint is running."}), 200


@app.post("/predict")
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
