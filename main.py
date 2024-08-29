import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

    df["processed_review_text"] = df["review_text"].apply(preprocess_text)

    print(df)


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


if __name__ == "__main__":
    main()
