import joblib
import numpy as np

svm_model = joblib.load('/Users/joycendichu/nlp_flask_app/models/svm_model.pkl')
vectorizer = joblib.load('/Users/joycendichu/nlp_flask_app/models/linear_svm_vectorizer.pkl')

test_review = "Your test review text here."
review_vector = vectorizer.transform([test_review])
prediction_probabilities = svm_model.predict_proba(review_vector)
prediction = svm_model.predict(review_vector)

print(f"Prediction probabilities: {prediction_probabilities}, Prediction: {prediction}")
