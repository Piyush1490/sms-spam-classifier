# SMS Spam Classifier - Clean & Production Ready

import numpy as np
import pandas as pd
import string
import pickle
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import MultinomialNB


# ---------------- SAFE NLTK DOWNLOAD ---------------- #
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


setup_nltk()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


# ---------------- TEXT PREPROCESSING ---------------- #
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    # keep alphanumeric
    words = [w for w in words if w.isalnum()]

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # stemming
    words = [ps.stem(w) for w in words]

    return " ".join(words)


# ---------------- LOAD & CLEAN DATA ---------------- #
def load_data(path="spam.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Keep dataset in project folder.")

    df = pd.read_csv(path, encoding='latin-1')

    # drop unnecessary columns (if present)
    cols_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # rename columns
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    # encode labels
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})

    # remove duplicates
    df = df.drop_duplicates(keep='first')

    return df


# ---------------- TRAIN MODEL ---------------- #
def train_model(df):
    df['transformed_text'] = df['text'].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")

    return model, tfidf


# ---------------- SAVE MODEL ---------------- #
def save_model(model, vectorizer):
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    print("\nModel saved as model.pkl & vectorizer.pkl")


# ---------------- LOAD MODEL ---------------- #
def load_model():
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    return None, None


# ---------------- PREDICTION ---------------- #
def predict_spam(text, model, vectorizer):
    transformed = transform_text(text)
    vector_input = vectorizer.transform([transformed]).toarray()
    prediction = model.predict(vector_input)[0]
    return "Spam" if prediction == 1 else "Ham"


# ---------------- MAIN ---------------- #
def main():
    print("=== SMS Spam Classifier ===")

    # Try loading existing model
    model, vectorizer = load_model()

    if model is None:
        print("Training new model...")
        df = load_data()
        model, vectorizer = train_model(df)
        save_model(model, vectorizer)
    else:
        print("Loaded existing model.")

    # Interactive loop
    while True:
        user_input = input("\nEnter message (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = predict_spam(user_input, model, vectorizer)
        print("Prediction:", result)


if __name__ == "__main__":
    main()