from typing import Optional
import nltk
import pandas as pd
from textblob import TextBlob

from models import DatasetConfig
from utils import preprocess_text


class SentimentAnalysis:
    config: Optional[DatasetConfig]
    df: pd.DataFrame

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        if config is None and df is None:
            raise ValueError("One of DatasetConfig or pd.DataFrame must be provided")

        self.config = config

        # download nltk resources
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")
        nltk.download("wordnet")

        # passing a DataFrame directly is meant for tests
        if df is not None:
            self.df = df
        elif config is not None:
            df = pd.read_csv(config.csv_path, nrows=config.max_rows)
            df = df[[config.review_text_column, config.rating_column]]
            df = df.rename(
                columns={
                    config.review_text_column: "review_text",
                    config.rating_column: "rating",
                }
            )
            self.df = df
        else:
            raise ValueError("Either config or df must be provided")

    def preprocess_reviews(self) -> pd.DataFrame:
        self.df["processed_review_text"] = self.df["review_text"].apply(preprocess_text)
        return self.df

    def basic_sentiment_analysis(self) -> pd.DataFrame:
        self.df["sentiment_score"] = self.df["processed_review_text"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df["sentiment"] = pd.cut(
            self.df["sentiment_score"],
            bins=[-1, -0.01, 0.1, 1],
            labels=["Negative", "Neutral", "Positive"],
        )
        return self.df

    def display_summary(self):
        sentiment_summary = self.df["sentiment"].value_counts().to_dict()
        print("Results:")
        for category, count in sentiment_summary.items():
            print(f"\t{category}: {count}")

    def save_results(self):
        if self.config is None:
            raise ValueError("Config is required to save results")

        # additional step to just show the columns we care most about first
        reordered_columns = ["rating", "sentiment", "sentiment_score"] + [
            col
            for col in self.df.columns
            if col not in ["rating", "sentiment", "sentiment_score"]
        ]
        self.df[reordered_columns].to_csv(self.config.output_csv_path, index=False)

        print(f"Output saved to: {self.config.output_csv_path}")
