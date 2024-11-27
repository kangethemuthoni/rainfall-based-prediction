from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from utils import process_disaster_record_for_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the saved model
lstm_model = load_model('models/mlp_model2.h5')

@app.route("/")
def home():
    return "Welcome to the Disaster Prediction Flask App!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate input
    if not all(key in data for key in ("Month", "Year", "Country")):
        return jsonify({"error": "Missing required fields: Month, Year, or Country"}), 400

    # Process the input
    record = {
        "Month": data["Month"],
        "Year": data["Year"],
        "Country": data["Country"]
    }
    features = process_disaster_record_for_model(record)

    # Predict using the model
    prediction = lstm_model.predict(features)
    confidence = float(prediction[0][0])

    return jsonify({
        "Month": record["Month"],
        "Year": record["Year"],
        "Country": record["Country"],
        "Prediction": "Drought" if confidence < 0.5 else "Normal",
        "Confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=7171)