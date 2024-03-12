# app.py
import os
import joblib
from flask import Flask, request, jsonify
from score import score

app = Flask(__name__)

# Load the model and vectorizer once at the start
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Request must contain a \"text\" field."}), 400

    text = data["text"]
    threshold = data.get("threshold", 0.5)  # Default threshold is 0.5
    prediction, propensity = score(text, model, vectorizer, threshold)
    return jsonify({"prediction": prediction, "propensity": propensity})

if __name__ == "__main__":
    # Set the FLASK_APP environment variable
    os.environ["FLASK_APP"] = "app"
    # Set the FLASK_ENV to development to enable debug mode
    os.environ["FLASK_ENV"] = "development"
    # Start the app
    app.run(port = 8576)
