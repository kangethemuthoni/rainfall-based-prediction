import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def train_logistic_regression_model(X_train, y_train):
    """
    Function to train the Logistic Regression model.
    """
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    return log_reg_model


def evaluate_model(log_reg_model, X_test, y_test):
    """
    Function to evaluate the Logistic Regression model on test data.
    """
    log_reg_predictions = log_reg_model.predict(X_test)
    accuracy = accuracy_score(y_test, log_reg_predictions)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, log_reg_predictions))
    return accuracy


def save_model(
    log_reg_model,
    vectorizer,
    model_path="/Users/joycendichu/nlp_flask_app/models/log_reg_model.pkl",
    vectorizer_path="/Users/joycendichu/nlp_flask_app/models/log_reg_vectorizer.pkl",
):
    """
    Function to save the trained Logistic Regression model and TF-IDF vectorizer to disk.
    """
    joblib.dump(log_reg_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def main():
    df = pd.read_csv("/Users/joycendichu/Downloads/clean_dataset.csv")
    X, y, tfidf_vectorizer = load_and_prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg_model = train_logistic_regression_model(X_train, y_train)

    evaluate_model(log_reg_model, X_test, y_test)

    save_model(
        log_reg_model,
        tfidf_vectorizer,
        model_path="/Users/joycendichu/nlp_flask_app/models/log_reg_model.pkl",
        vectorizer_path="/Users/joycendichu/nlp_flask_app/models/log_reg_vectorizer.pkl",
    )


if __name__ == "__main__":
    main()
