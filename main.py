import pandas as pd
import nltk
from textblob import TextBlob

from models import DatasetConfig
from utils import extract_features, preprocess_text


def download_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
    for resource in resources:
        nltk.download(resource)


def load_and_prepare_data(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, nrows=config.max_rows)
    df = df[[config.review_text_column, config.rating_column]]
    df = df.rename(
        columns={
            config.review_text_column: "review_text",
            config.rating_column: "rating",
        }
    )
    return df


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df["processed_review_text"] = df["review_text"].apply(preprocess_text)
    return df


def basic_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment_score"] = df["processed_review_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["sentiment"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


def display_summary(df: pd.DataFrame):
    sentiment_summary = df["sentiment"].value_counts().to_dict()
    print("Results:")
    for category, count in sentiment_summary.items():
        print(f"\t{category}: {count}")


def save_results(df: pd.DataFrame, output_path: str):
    reordered_columns = ["rating", "sentiment", "sentiment_score"] + [
        col
        for col in df.columns
        if col not in ["rating", "sentiment", "sentiment_score"]
    ]
    df[reordered_columns].to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    download_nltk_resources()

    config = DatasetConfig(
        csv_path="./product_reviews_data.csv",
        review_text_column="reviews.text",
        rating_column="reviews.rating",
        max_rows=200,
        output_csv_path="output.csv",
    )

    df = load_and_prepare_data(config)
    df = preprocess_reviews(df)
    df = basic_sentiment_analysis(df)

    extract_features(df)

    # optional step just for visualizing
    # from utils import visualize
    # visualize(df)

    display_summary(df)
    save_results(df, config.output_csv_path)
