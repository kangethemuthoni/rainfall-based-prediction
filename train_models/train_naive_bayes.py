import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_and_prepare_data(df):
    """
    Function to prepare the data for training using TF-IDF vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df["cleaned_review"])
    y = df["sentiment"].values
    return X, y, tfidf_vectorizer


def train_naive_bayes_model(X_train, y_train):
    """
    Function to train the Naive Bayes model.
    """
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model


def evaluate_model(nb_model, X_test, y_test):
    """
    Function to evaluate the Naive Bayes model on test data.
    """
    nb_predictions = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, nb_predictions)
    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, nb_predictions))
    return accuracy


def save_model(
    nb_model,
    vectorizer,
    model_path="models/naive_bayes_model.pkl",
    vectorizer_path="models/naive_bayes_vectorizer.pkl",
):
    """
    Function to save the trained Naive Bayes model and TF-IDF vectorizer to disk.
    """
    joblib.dump(nb_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def main():
    df = pd.read_csv(
        ""
    )  # Replace with your actual data

    X, y, tfidf_vectorizer = load_and_prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nb_model = train_naive_bayes_model(X_train, y_train)

    evaluate_model(nb_model, X_test, y_test)

    save_model(
        nb_model,
        tfidf_vectorizer,
        model_path="models/naive_bayes_model.pkl",
        vectorizer_path="models/naive_bayes_vectorizer.pkl",
    )


if __name__ == "__main__":
    main()
