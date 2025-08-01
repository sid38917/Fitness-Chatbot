from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pandas as pd


def train_sentiment_model():
    #sentiment files
    label_dir = {
        "positive": "positive_sentiment.csv",
        "negative": "negative_sentiment.csv"
    }
    data = []
    labels = []


    for label, path in label_dir.items():
        df = pd.read_csv(path)
        data.extend(df['Text'])
        labels.extend([label] * len(df))

    #trains a sentiment classifier
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    #transforms the text into a bag of words representation
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(X_train)

    #converts the bag of words counters into TF-IDF scores to emphasize the importance of less common but meaningful words
    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    classifier = LogisticRegression()
    classifier.fit(X_train_tfidf, y_train)


    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = classifier.predict(X_test_tfidf)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return classifier, count_vect, tfidf_transformer

def is_sentiment_related(user_input, keywords):
    return any(keyword in user_input.lower() for keyword in keywords)