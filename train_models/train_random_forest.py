import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
import joblib

def load_and_prepare_data(df):
    """
    Function to load, sample, and prepare data for training.
    """
    # Sampling the dataset (adjust the frac as needed)
    df_sampled = df.sample(frac=0.1, random_state=42)
    
    # Vectorizing the text features
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(df_sampled['cleaned_review'])

    # Extracting and one-hot encoding other features
    X_other_features = df_sampled.drop(columns=['cleaned_review', 'adjectives_adverbs', 'pos_tagged_review', 'sentiment'])
    X_other_features = pd.get_dummies(X_other_features, drop_first=True)
    X_other_features = csr_matrix(X_other_features.values)

    # Combining text and other features
    X = hstack([X_text, X_other_features])
    y = df_sampled['sentiment']

    return X, y, vectorizer

def train_model(X_train, y_train):
    """
    Function to train the RandomForest model.
    """
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Function to evaluate the trained model on test data.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def save_model(clf, vectorizer, model_path='/Users/joycendichu/nlp_flask_app/models/random_forest.pkl', vectorizer_path='/Users/joycendichu/nlp_flask_app/models/random_forest_vectorizer.pkl'):
    """
    Function to save the trained model and vectorizer to disk.
    """
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f'Model saved to {model_path}')
    print(f'Vectorizer saved to {vectorizer_path}')

def main():
    df = pd.read_csv('/Users/joycendichu/Downloads/clean_dataset.csv')  # Load your cleaned dataset here
    X, y, vectorizer = load_and_prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = train_model(X_train, y_train)

    evaluate_model(clf, X_test, y_test)

    save_model(clf, vectorizer, model_path='/Users/joycendichu/nlp_flask_app/models/random_forest.pkl', vectorizer_path='/Users/joycendichu/nlp_flask_app/models/random_forest_vectorizer.pkl')

if __name__ == "__main__":
    main()
