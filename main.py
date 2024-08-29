import pandas as pd
from models import DatasetConfig

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

print(df)
