import pandas as pd
import pytest
from utils import extract_features, preprocess_text


@pytest.fixture
def sample_df():
    data = {
        "review_text": [
            "This product is amazing and fantastic!",
            "I love this item, it's great and awesome.",
            "Terrible product, very disappointing.",
            "Awful experience, would not recommend.",
        ],
        "rating": [5, 5, 1, 1],
        "expected_positive_keywords": ["amazing", "fantastic", "great", "awesome"],
        "expected_negative_keywords": ["terrible", "disappointing", "awful"],
        "expected_positive_phrases": [
            "product amazing",
            "amazing fantastic",
            "great awesome",
        ],
        "expected_negative_phrases": ["terrible product", "awful experience"],
    }
    df = pd.DataFrame(data)
    df["processed_review_text"] = df["review_text"].apply(preprocess_text)
    df["sentiment"] = ["Positive", "Positive", "Negative", "Negative"]
    return df


def test_extract_features(sample_df):
    features = extract_features(sample_df)

    for positive_keyword in sample_df["expected_positive_keywords"].iloc[0]:
        assert positive_keyword in features["top_positive_keywords"]

    for negative_keyword in sample_df["expected_negative_keywords"].iloc[0]:
        assert negative_keyword in features["top_negative_keywords"]

    assert any(
        positive_phrase in " ".join(features["top_positive_phrases"])
        for positive_phrase in sample_df["expected_positive_phrases"].iloc[0]
    )

    for negative_phrase in sample_df["expected_negative_phrases"].iloc[0]:
        assert negative_phrase in features["top_negative_phrases"]
