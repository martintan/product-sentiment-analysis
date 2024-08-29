import re
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd


"""
Preprocess text by removing punctuation, stop words, and lemmatizing.
Apply this to each review in the dataset.

@param text: the text to preprocess
@return: the preprocessed text
"""


def preprocess_text(text: str):
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


"""
Visualize sentiment analysis results.
Note: matplotlib is only imported if this function is called.

@param df: the dataframe to visualize
"""


def visualize(df: pd.DataFrame):
    # only import matplotlib if visualize is actually called
    import matplotlib.pyplot as plt

    # diagram 1: sentiment analysis results (bar plot)
    sentiment_counts = df["sentiment"].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind="bar")
    plt.title("sentiment analysis results")
    plt.xlabel("sentiment")
    plt.ylabel("# of reviews")
    plt.tight_layout()
    plt.show()

    # diagram 2: sentiment vs. rating (scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["rating"], df["sentiment_score"])
    plt.title("sentiment vs. rating")
    plt.xlabel("rating")
    plt.ylabel("sentiment score")
    plt.tight_layout()
    plt.show()


"""
Get the top tokens / n-grams from a text series.

@param text_series: the text series to extract tokens from
@param num_words: the number of words in the n-grams
@param max_keywords: the maximum number of keywords to return
"""


def get_top_tokens(text_series: pd.Series, num_words: int, max_keywords: int = 10):
    all_ngrams = [
        gram for text in text_series for gram in ngrams(text.split(), num_words)
    ]
    return sorted(Counter(all_ngrams).items(), key=lambda x: x[1], reverse=True)[
        :max_keywords
    ]


"""
Extract top keywords & phrases from the dataset.

@param df: the dataframe containing the dataset
@return: A dictionary containing the extracted features
"""


def extract_features(df: pd.DataFrame):
    positive_reviews = df[df["sentiment"] == "Positive"]["processed_review_text"]
    negative_reviews = df[df["sentiment"] == "Negative"]["processed_review_text"]

    positive_keywords = get_top_tokens(positive_reviews, 1)
    positive_phrases = get_top_tokens(positive_reviews, 2)

    negative_keywords = get_top_tokens(negative_reviews, 1)
    negative_phrases = get_top_tokens(negative_reviews, 2)

    def format_tokens(tokens, is_phrase=False):
        return [" ".join(token[0]) if is_phrase else token[0][0] for token in tokens]

    return {
        "top_positive_keywords": format_tokens(positive_keywords),
        "top_positive_phrases": format_tokens(positive_phrases, is_phrase=True),
        "top_negative_keywords": format_tokens(negative_keywords),
        "top_negative_phrases": format_tokens(negative_phrases, is_phrase=True),
    }
