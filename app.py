from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
from utils import preprocess_text  # Assuming you have a preprocess_text function
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model_files = {
    'logistic_regression': ('models/logistic_regression.pkl', 'models/logistic_regression_vectorizer.pkl'),
    'random_forest': ('models/random_forest.pkl', 'models/random_forest_vectorizer.pkl'),
    'svm': ('models/svm.pkl', 'models/svm_vectorizer.pkl'),
    'naive_bayes': ('models/naive_bayes.pkl', 'models/naive_bayes_vectorizer.pkl')
}

models = {}
vectorizers = {}

for model_name, (model_path, vectorizer_path) in model_files.items():
    with open(model_path, 'rb') as model_file:
        models[model_name] = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizers[model_name] = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    review = data['review']

    # Preprocess the review
    cleaned_review = preprocess_text(review)

    # Load non-text feature columns (if you have any other features besides text)
    # Assuming you have a dictionary of other features from the request data
    non_text_features = data.get('other_features', {})
    non_text_df = pd.DataFrame([non_text_features])

    # Load the saved feature column names from training
    with open('feature_columns.pkl', 'rb') as columns_file:
        feature_columns = pickle.load(columns_file)

    # Align non-text features with the saved columns (fill missing with 0)
    X_other_features_inference = pd.get_dummies(non_text_df, drop_first=True)
    X_other_features_inference = X_other_features_inference.reindex(columns=feature_columns, fill_value=0)
    X_other_features_inference = csr_matrix(X_other_features_inference.values)  # Convert to sparse matrix

    best_model = None
    best_prediction = None
    highest_confidence = -1  # Start with the lowest possible confidence

    for model_name, model in models.items():
        vectorizer = vectorizers[model_name]

        try:
            # Vectorize the cleaned review text
            review_vector = vectorizer.transform([cleaned_review])

            # Combine the text features with the non-text features (stacking them)
            X_inference = hstack([review_vector, X_other_features_inference])

            if hasattr(model, 'predict_proba'):
                # For models that provide probability predictions (like Logistic Regression, Random Forest)
                prediction_probabilities = model.predict_proba(X_inference)
                confidence = max(prediction_probabilities[0])  # Get the highest probability
                prediction = model.predict(X_inference)
            else:
                # For models that do not support probabilities (like SVM without calibration)
                prediction = model.predict(X_inference)
                confidence = 1  # Assume 100% confidence for models without probabilities

            # Track the best prediction (highest confidence)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_model = model_name
                best_prediction = prediction[0]  # Get the actual label

        except ValueError as e:
            return jsonify({"error": f"Model {model_name} encountered an error: {str(e)}"})

    # Return the best prediction with model name and confidence
    return jsonify({
        "sentiment": best_prediction,
        "model": best_model,
        "confidence": highest_confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=7070)