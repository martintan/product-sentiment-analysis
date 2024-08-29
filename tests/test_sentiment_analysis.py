import pandas as pd
import pytest
from sentiment_analysis import SentimentAnalysis

"""
I know this way of generating sample data is not ideal, but it's a quick way for testing.
"""


@pytest.fixture
def sample_data():
    positive_reviews = [
        "This product is amazing! I love it.",
        "Excellent quality and great customer service.",
        "Highly recommended. Will buy again.",
        "Exceeded my expectations. Fantastic purchase.",
        "Best product I've ever used. Simply wonderful.",
    ] * 5

    negative_reviews = [
        "Terrible product. Don't waste your money.",
        "Poor quality and awful customer service.",
        "Disappointed with this purchase. Avoid it.",
        "Broke after a week. Very frustrating experience.",
        "Worst product ever. Complete waste of time.",
    ] * 5

    all_reviews = positive_reviews + negative_reviews
    return pd.DataFrame({"review_text": all_reviews, "rating": [5] * 25 + [1] * 25})


def test_sentiment_analysis(sample_data):
    sa = SentimentAnalysis(df=sample_data)
    sa.preprocess_reviews()
    sa.basic_sentiment_analysis()

    positive_count = sa.df[sa.df["sentiment"] == "Positive"].shape[0]
    negative_count = sa.df[sa.df["sentiment"] == "Negative"].shape[0]
    assert positive_count == 25, f"Expected 25 positive reviews, but got {positive_count}"
    assert negative_count == 25, f"Expected 25 negative reviews, but got {negative_count}"

    high_rating_positive = sa.df[(sa.df["rating"] == 5) & (sa.df["sentiment"] == "Positive")].shape[0]
    low_rating_negative = sa.df[(sa.df["rating"] == 1) & (sa.df["sentiment"] == "Negative")].shape[0]
    assert high_rating_positive == 25, f"Expected 25 high-rated positive reviews, but got {high_rating_positive}"
    assert low_rating_negative == 25, f"Expected 25 low-rated negative reviews, but got {low_rating_negative}"
