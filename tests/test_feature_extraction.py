import pandas as pd
import pytest
from sentiment_analysis import SentimentAnalysis
from utils import extract_features, preprocess_text


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "review_text": [
                "This product is amazing and fantastic!",
                "I love this item, it's great and awesome.",
                "Terrible product, very disappointing.",
                "This product is a complete waste of money.",
            ],
            "rating": [5, 5, 1, 1],
        }
    )


@pytest.fixture
def expected_data():
    return {
        "expected_positive_keywords": ["amazing", "fantastic", "great", "awesome"],
        "expected_negative_keywords": ["terrible", "disappointing", "waste"],
        "expected_positive_phrases": [
            "product amazing",
            "amazing fantastic",
            "great awesome",
        ],
        "expected_negative_phrases": ["terrible product", "complete waste"],
    }


@pytest.fixture
def sentiment_analysis(sample_df):
    sa = SentimentAnalysis(df=sample_df)
    sa.preprocess_reviews()
    sa.basic_sentiment_analysis()
    return sa


def test_extract_features(sentiment_analysis, expected_data):
    features = extract_features(sentiment_analysis.df)

    print(features)

    for positive_keyword in expected_data["expected_positive_keywords"]:
        assert positive_keyword in features["top_positive_keywords"]

    for negative_keyword in expected_data["expected_negative_keywords"]:
        assert negative_keyword in features["top_negative_keywords"]

    assert any(
        positive_phrase in " ".join(features["top_positive_phrases"])
        for positive_phrase in expected_data["expected_positive_phrases"]
    )

    for negative_phrase in expected_data["expected_negative_phrases"]:
        assert negative_phrase in features["top_negative_phrases"]
