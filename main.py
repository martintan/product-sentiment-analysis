import pandas as pd
import nltk
from textblob import TextBlob
import nltk

from models import DatasetConfig
from utils import extract_features, preprocess_text


def sentiment_analysis():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")

    # configure the dataset here
    config = DatasetConfig(
        csv_path="./product_reviews_data.csv",
        review_text_column="reviews.text",
        rating_column="reviews.rating",
        max_rows=200,
        output_csv_path="output.csv",
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
    df["sentiment_score"] = df["processed_review_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["sentiment"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["Negative", "Neutral", "Positive"],
    )

    extract_features(df)

    # from utils import visualize
    # visualize(df)

    # display summary
    sentiment_summary = df["sentiment"].value_counts().to_dict()
    print("Results:")
    for category, count in sentiment_summary.items():
        print(f"\t{category}: {count}")

    # reorder columns for readability
    reordered_columns = ["rating", "sentiment", "sentiment_score"] + [
        col
        for col in df.columns
        if col not in ["rating", "sentiment", "sentiment_score"]
    ]
    # save processed data to csv
    df[reordered_columns].to_csv(config.output_csv_path, index=False)
    print(f"Output saved to: {config.output_csv_path}")


if __name__ == "__main__":
    sentiment_analysis()
