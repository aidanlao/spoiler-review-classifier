import gzip
import json
import random

import pandas as pd
import numpy as np

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

    # Assign spoiler label to review that contains "(view spoiler)"
    df['has_spoiler'] = np.where(df['review_text'].str.contains("(view spoiler)", regex=False), 1, 0)

    # All spoilers contain text of form "(view spoiler)[... (hide spoiler)]" so remove the (view spoiler) and (hide spoiler) part
    df['review_text'] = df["review_text"].str.replace("(view spoiler)[", "", regex=False)
    df['review_text'] = df["review_text"].str.replace("(hide spoiler)]", "", regex=False)

    return df

if __name__=="__main__":
    N = 10000
    PATH_TO_DATASET = "goodreads_reviews_spoiler_raw.json.gz"
    KEEP_COLS = ['book_id', 'review_text']
    SAVE_TO_CSV = "goodreads_reviews.csv"

    df = reservoir_sampling(PATH_TO_DATASET, N)
    df = clean_dataset(df, KEEP_COLS)
    df.to_csv(SAVE_TO_CSV, index=False)

# Since data has book_id we can avoid capturing multiple reviews on the same book else the model will train on book specific information (Data leakage consideration)
# Split the dataset into 80/10/10, so multiple csv files
