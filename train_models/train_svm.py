# train_svm.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
new_df = pd.read_csv('/Users/joycendichu/Downloads/new_df.csv')

# Check the DataFrame shape and for missing values
print("DataFrame shape:", new_df.shape)
print("Missing values:\n", new_df.isnull().sum())

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['cleaned_review'])

# Display the shapes of the TF-IDF matrix and sentiment labels
print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Sentiment shape:", new_df['sentiment'].shape)

# Prepare the target variable
y = new_df['sentiment'].values.ravel()

# Check for consistent number of samples
if tfidf_matrix.shape[0] != y.shape[0]:
    raise ValueError("Inconsistent number of samples: TF-IDF has {} samples while sentiment has {} samples.".format(tfidf_matrix.shape[0], y.shape[0]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# Initialize and train the Linear SVC model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(X_test)

# Evaluate the model's performance
print("Linear SVM Accuracy: ", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

# Save the trained model to a file
with open('/Users/joycendichu/nlp_flask_app/models/svm.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

# Save the TF-IDF vectorizer to a file
with open('/Users/joycendichu/nlp_flask_app/models/svm_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
