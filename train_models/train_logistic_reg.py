import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

new_df = pd.read_csv('/Users/joycendichu/Downloads/new_df.csv')

tfidf_vectorizer = TfidfVectorizer()

X = tfidf_vectorizer.fit_transform(new_df['cleaned_review'])

y = new_df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=1000)

log_reg_model.fit(X_train, y_train)

log_reg_predictions = log_reg_model.predict(X_test)

print("Logistic Regression Accuracy: ", accuracy_score(y_test, log_reg_predictions))
print(classification_report(y_test, log_reg_predictions))

with open('/Users/joycendichu/nlp_flask_app/models/logistic_regression.pkl', 'wb') as model_file:
    pickle.dump(log_reg_model, model_file)

with open('/Users/joycendichu/nlp_flask_app/models/logistic_regression_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
