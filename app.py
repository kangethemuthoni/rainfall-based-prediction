import numpy as np
from flask import Flask, request, jsonify
import joblib
from utils import clean_process_and_extract_features
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"

# Define the custom MSE loss function (if applicable to any deep learning models)
def custom_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# Define your model and vectorizer paths
model_files = {
    "cnn": ("models/cnn.h5", "models/tfidf_vectorizer.pkl"),
    "gradient_boosting": ("models/gradient_boosting.pkl", "models/tfidf_vectorizer.pkl"),
    "logistic_regression": ("models/logistic_regression.pkl", "models/tfidf_vectorizer.pkl"),
    "mlp": ("models/mlp.h5", "models/tfidf_vectorizer.pkl"),
    "random_forest": ("models/random_forest.pkl", "models/tfidf_vectorizer.pkl"),
    "svm": ("models/svm.pkl", "models/tfidf_vectorizer.pkl")
}

# Load models and TF-IDF vectorizer
models = {}
tfidf_vectorizers = {}
for model_name, (model_path, vectorizer_path) in model_files.items():
    try:
        # Load deep learning models (e.g., CNN, MLP)
        if model_path.endswith(".h5"):
            models[model_name] = load_model(model_path, custom_objects={'custom_mse_loss': custom_mse_loss})
        else:
            # Load classical ML models
            models[model_name] = joblib.load(model_path)

        # Load the vectorizer
        if vectorizer_path:
            tfidf_vectorizers[model_name] = joblib.load(vectorizer_path)

        print(f"{model_name} model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

@app.route("/")
def home():
    return "Welcome to the AI Prediction API!"

@app.route("/predict", methods=["POST","GET"])
def predict():
    IF
    try:
        # Parse the request data
        data = request.get_json()
        biography = data["biography"]

        # Preprocess the user input
        cleaned_input = clean_process_and_extract_features(biography)  # Apply text cleaning
        cleaned_input_tfidf = tfidf_vectorizers["logistic_regression"].transform([cleaned_input])  # Vectorize input

        # Prepare combined features if necessary (e.g., TF-IDF + additional features)
        # Add any additional feature engineering here if applicable

        # Iterate through models and make predictions
        best_model = None
        best_prediction = None
        highest_confidence = -1
        confidence_scores = {}

        for model_name, model in models.items():
            print(f"Making prediction with {model_name}...")
            
            if model_name in ["cnn", "mlp"]:
                # Reshape the input for Keras models if necessary
                reshaped_input = np.array(cleaned_input_tfidf.toarray()).reshape((1, -1))
                prediction = model.predict(reshaped_input)
                confidence = np.max(prediction)  # Take the highest probability
                prediction = np.argmax(prediction)  # Get the class index
            elif model_name == "svm":
                # SVM-specific decision function
                decision_score = model.decision_function(cleaned_input_tfidf)
                confidence = 1 / (1 + np.exp(-decision_score[0]))  # Sigmoid for confidence
                prediction = model.predict(cleaned_input_tfidf)
            elif hasattr(model, "predict_proba"):
                # Models with `predict_proba` (e.g., Logistic Regression, Random Forest, etc.)
                prediction_probabilities = model.predict_proba(cleaned_input_tfidf)
                confidence = max(prediction_probabilities[0])  # Get the highest probability
                prediction = model.predict(cleaned_input_tfidf)
            else:
                # Models without probability output
                prediction = model.predict(cleaned_input_tfidf)
                confidence = 1  # Default confidence for non-probabilistic models

            # Store the confidence score
            confidence_scores[model_name] = confidence
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_prediction = prediction[0] if isinstance(prediction, np.ndarray) else prediction

        # Return the best prediction
        return jsonify({
            "input": biography,
            "prediction": int(best_prediction),  # Ensure it's JSON serializable
            "confidence": float(highest_confidence),
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=7070)







