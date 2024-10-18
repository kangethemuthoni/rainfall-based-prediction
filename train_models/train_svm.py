import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare_data(df):
    """
    Function to prepare data for training.
    """
    # Vectorizing the text features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review'])

    y = df['sentiment'].values.ravel()

    # Ensure the shapes match
    if tfidf_matrix.shape[0] != y.shape[0]:
        raise ValueError(f"Inconsistent number of samples: TF-IDF has {tfidf_matrix.shape[0]} samples while sentiment has {y.shape[0]} samples.")

    return tfidf_matrix, y, tfidf_vectorizer

def train_svm_model(X_train, y_train):
    """
    Function to train the SVM model.
    """
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(svm_model, X_test, y_test):
    """
    Function to evaluate the SVM model on test data.
    """
    svm_predictions = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, svm_predictions)
    print(f"Linear SVM Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, svm_predictions))
    return accuracy

def save_model(svm_model, vectorizer, model_path='/Users/joycendichu/nlp_flask_app/models/svm_model.pkl', vectorizer_path='/Users/joycendichu/nlp_flask_app/models/svm_vectorizer.pkl'):
    """
    Function to save the trained SVM model and TF-IDF vectorizer to disk.
    """
    joblib.dump(svm_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f'Model saved to {model_path}')
    print(f'Vectorizer saved to {vectorizer_path}')

def main():
    df = pd.read_csv('/Users/joycendichu/Downloads/clean_dataset.csv')  
    
    X, y, tfidf_vectorizer = load_and_prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = train_svm_model(X_train, y_train)

    evaluate_model(svm_model, X_test, y_test)

    save_model(svm_model, tfidf_vectorizer, model_path='/Users/joycendichu/nlp_flask_app/models/svm_model.pkl', vectorizer_path='/Users/joycendichu/nlp_flask_app/models/svm_vectorizer.pkl')

if __name__ == "__main__":
    main()
