import gzip
import json
import random

import pandas as pd
import numpy as np

from get_dataset_info import get_dataset_info

"""
Reads a compressed JSON file and uses reservoir sampling to randomly select n reviews from the original dataset.
Because of the ~1m entries in the orginal dataset and our need for only a subset of it, reservoir sampling
allows us to pull a subset while providing equal chance for every entry to be included.
"""
def reservoir_sampling(path: str, n: int) -> pd.DataFrame:
    reviews = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            review = json.loads(line)

            if i < n:
                reviews.append(review)
            else:
                j = random.randint(0, i)

                if j < n:
                    reviews[j] = review

    return pd.DataFrame(reviews)

def clean_dataset(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    # Remove useless features
    df = df[keep_cols].copy()

    # Assign spoiler label to review that contains "(view spoiler)" and to reviews without
    df['has_spoiler'] = np.where(df['review_text'].str.contains("(view spoiler)", regex=False), 1, 0)

    # All spoilers contain text of form "(view spoiler)[... (hide spoiler)]" so remove the (view spoiler) and (hide spoiler) part
    df['review_text'] = df["review_text"].str.replace("(view spoiler)[", "", regex=False)
    df['review_text'] = df["review_text"].str.replace("(hide spoiler)]", "", regex=False)

    return df

def split_dataset(df: pd.DataFrame, csv_name: str, training_size: int, validation_size: int, testing_size: int) -> pd.DataFrame:
    train_slice = int(len(df) * training_size)
    val_slice = train_slice + int(len(df) * validation_size)
    test_slice = val_slice + int(len(df) * testing_size)

    training_data = df.iloc[:train_slice]
    validation_data = df.iloc[train_slice:val_slice]
    testing_data = df.iloc[val_slice:test_slice]

    training_data.to_csv(f"training_{csv_name}", index=False)
    validation_data.to_csv(f"validation_{csv_name}", index=False)
    testing_data.to_csv(f"testing_{csv_name}", index=False)

    print("Training Dataset:")
    get_dataset_info(training_data)
    print("Validation Dataset:")
    get_dataset_info(validation_data)
    print("Testing Dataset:")
    get_dataset_info(testing_data)

if __name__=="__main__":
    N = 10000
    PATH_TO_DATASET = "goodreads_reviews_spoiler_raw.json.gz"
    KEEP_COLS = ['book_id', 'review_text']
    SAVE_TO_CSV = "goodreads_reviews.csv"

    df = reservoir_sampling(PATH_TO_DATASET, N)
    df = clean_dataset(df, KEEP_COLS)
    df.to_csv(SAVE_TO_CSV)
    split_dataset(df, SAVE_TO_CSV, 0.8, 0.1, 0.1)
