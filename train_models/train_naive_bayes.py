import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
new_df = pd.read_csv('/Users/joycendichu/Downloads/new_df.csv')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transform the cleaned review text into TF-IDF features
X = tfidf_vectorizer.fit_transform(new_df['cleaned_review'])

# Target variable
y = new_df['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
nb_model = MultinomialNB()

# Fit the model to the training data
nb_model.fit(X_train, y_train)

# Make predictions on the test set
nb_predictions = nb_model.predict(X_test)

# Evaluate the model's performance
print("Naive Bayes Accuracy: ", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

# Save the trained model
with open('/Users/joycendichu/nlp_flask_app/models/naive_bayes.pkl', 'wb') as model_file:
    pickle.dump(nb_model, model_file)

# Save the TF-IDF vectorizer
with open('/Users/joycendichu/nlp_flask_app/models/naive_bayes_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
