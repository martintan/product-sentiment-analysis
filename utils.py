from nltk.util import ngrams
from collections import Counter
import pandas as pd

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
