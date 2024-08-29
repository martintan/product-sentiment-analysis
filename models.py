from dataclasses import dataclass


# we don't need pydantic yet
@dataclass
class DatasetConfig:
    csv_path: str
    review_text_column: str
    rating_column: str
    max_rows: int = 100
