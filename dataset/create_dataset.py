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

            # Initially fill list with n reviews
            if i < n:
                reviews.append(review)
            else: # After list is full, randomly replace reviews with other reviews
                j = random.randint(0, i)

                if j < n:
                    reviews[j] = review

    return pd.DataFrame(reviews)

def stratified_sampling(path: str, n: int, spoiler_size: int, non_spoiler_size: int):
    spoilers = []
    non_spoilers = []
    len_sp = int(n * spoiler_size)
    len_np = int(n * non_spoiler_size)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            review = json.loads(line)

            # Reviews that contain spoilers
            if  '(view spoiler)' in review['review_text']:
                # Initially fill list
                if len(spoilers) < len_sp:
                    spoilers.append(review)
                else: # After list is filled, randomly replace reviews with spoilers
                    j = random.randint(0, len_sp - 1)

                    if j < n:
                        spoilers[j] = review

            # Reviews that don't contain spoilers
            else:
                # Initially fill list
                if len(non_spoilers) < len_np:
                    non_spoilers.append(review)
                else: # After list is filled, randomly replace reviews with non spoilers
                    j = random.randint(0, len_np - 1)

                    if j < n:
                        non_spoilers[j] = review

    # Combine into one dataframe and randomize the rows
    reviews = spoilers + non_spoilers
    df = pd.DataFrame(reviews)
    df = df.sample(frac=1, random_state=50).reset_index(drop=True)

    return df

def clean_dataset(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    # Remove useless features
    df = df[keep_cols].copy()

    # Assign spoiler label to review that contains "(view spoiler)" and to reviews without
    df['has_spoiler'] = np.where(df['review_text'].str.contains("(view spoiler)", regex=False), 1, 0)

    # All spoilers contain text of form "(view spoiler)[... (hide spoiler)]" so remove the (view spoiler) and (hide spoiler) part
    df['review_text'] = df["review_text"].str.replace("(view spoiler)[", "", regex=False)
    df['review_text'] = df["review_text"].str.replace("(hide spoiler)]", "", regex=False)

    return df

def split_dataset(df: pd.DataFrame, csv_name: str, training_size: int, validation_size: int, testing_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_slice = int(len(df) * training_size)
    val_slice = train_slice + int(len(df) * validation_size)
    test_slice = val_slice + int(len(df) * testing_size)

    training_data = df.iloc[:train_slice]
    validation_data = df.iloc[train_slice:val_slice]
    testing_data = df.iloc[val_slice:test_slice]

    training_data.to_csv(f"training_{csv_name}", index=False)
    validation_data.to_csv(f"validation_{csv_name}", index=False)
    testing_data.to_csv(f"testing_{csv_name}", index=False)

    return training_data, validation_data, testing_data

if __name__=="__main__":
    N = 10000
    PATH_TO_DATASET = "goodreads_reviews_spoiler_raw.json.gz"
    KEEP_COLS = ['review_id', 'book_id', 'review_text']
    SAVE_TO_CSV = "goodreads_reviews.csv"

    random.seed(100)

    # df = reservoir_sampling(PATH_TO_DATASET, N)
    df = stratified_sampling(PATH_TO_DATASET, N, 0.2, 0.8)
    df = clean_dataset(df, KEEP_COLS)
    df.to_csv(SAVE_TO_CSV)
    training_data, validation_data, testing_data = split_dataset(df, SAVE_TO_CSV, 0.8, 0.1, 0.1)

    print("Original Dataset:")
    get_dataset_info(df)
    print("Training Dataset:")
    get_dataset_info(training_data)
    print("Validation Dataset:")
    get_dataset_info(validation_data)
    print("Testing Dataset:")
    get_dataset_info(testing_data)


