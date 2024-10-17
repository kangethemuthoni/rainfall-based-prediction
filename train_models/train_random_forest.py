import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

# Load your data
new_df = pd.read_csv('/Users/joycendichu/Downloads/new_df.csv')

# Take a 10% sample of the data
df_sampled = new_df.sample(frac=0.1, random_state=42)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Prepare text features
X_text = vectorizer.fit_transform(df_sampled['cleaned_review'])

# Prepare other features (excluding certain columns)
X_other_features = df_sampled.drop(columns=['cleaned_review', 'adjectives_adverbs', 'pos_tagged_review', 'sentiment'])

# Convert categorical variables to dummy variables
X_other_features = pd.get_dummies(X_other_features, drop_first=True)

# Convert to sparse matrix
X_other_features = csr_matrix(X_other_features.values)

# Combine text and other features
X = hstack([X_text, X_other_features])

# Target variable
y = df_sampled['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
model_path = '/Users/joycendichu/nlp_flask_app/models/random_forest.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Save the vectorizer
vectorizer_path = '/Users/joycendichu/nlp_flask_app/models/random_forest_vectorizer.pkl'
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Save the feature column names for consistency during inference
feature_columns_path = '/Users/joycendichu/nlp_flask_app/models/feature_columns.pkl'
with open(feature_columns_path, 'wb') as columns_file:
    pickle.dump(X_other_features.columns.tolist(), columns_file)

print("Model, vectorizer, and feature columns saved successfully.")
