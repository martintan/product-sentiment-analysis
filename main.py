import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from models import DatasetConfig


def main():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")

    # configure the dataset here
    config = DatasetConfig(
        csv_path="./product_reviews_data.csv",
        review_text_column="reviews.text",
        rating_column="reviews.rating",
        max_rows=100,
    )

    df = pd.read_csv(config.csv_path, nrows=config.max_rows)
    # only get the relevant columns
    df = df[[config.review_text_column, config.rating_column]]
    # just make sure the column names are consistent
    df = df.rename(
        columns={
            config.review_text_column: "review_text",
            config.rating_column: "rating",
        }
    )

    # preprocessing
    df["processed_review_text"] = df["review_text"].apply(preprocess_text)

    # sentiment analysis
    df["sentiment"] = df["processed_review_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["sentiment_category"] = pd.cut(
        df["sentiment"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["Negative", "Neutral", "Positive"],
    )

    visualize(df)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def visualize(df):
    # only import matplotlib if visualize is actually called
    import matplotlib.pyplot as plt

    # diagram 1: sentiment analysis results (bar plot)
    sentiment_counts = df['sentiment_category'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('sentiment analysis results')
    plt.xlabel('sentiment')
    plt.ylabel('# of reviews')
    plt.tight_layout()
    plt.show()

    # diagram 2: sentiment vs. rating (scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['rating'], df['sentiment'])
    plt.title('sentiment vs. rating')
    plt.xlabel('rating')
    plt.ylabel('sentiment')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
