import gzip
import json
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

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

"""
Use stratified sampling instead of reservoir sampling. Similar to reservoir sampling but allows us to forcefully adjust
the number of spoilers and non-spoilers within the dataset. Pass spoiler size = 0.2 and non_spoiler_size = 0.8 and the
dataset will consist of 20% spoilers and 80% non-spoilers.
"""
def stratified_sampling(path: str, n: int, spoiler_size: int, non_spoiler_size: int):
    spoilers = []
    non_spoilers = []
    len_sp = int(n * spoiler_size)
    len_np = int(n * non_spoiler_size)
    spoiler_count = 0
    non_spoiler_count = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            review = json.loads(line)

            # Reviews that contain spoilers
            if  '(view spoiler)' in review['review_text']:
                spoiler_count += 1
                # Initially fill list
                if len(spoilers) < len_sp:
                    spoilers.append(review)
                else: # After list is filled, randomly replace reviews with spoilers
                    j = random.randint(0, spoiler_count - 1)

                    if j < len_sp:
                        spoilers[j] = review

            # Reviews that don't contain spoilers
            else:
                non_spoiler_count += 1
                # Initially fill list
                if len(non_spoilers) < len_np:
                    non_spoilers.append(review)
                else: # After list is filled, randomly replace reviews with non spoilers
                    j = random.randint(0, non_spoiler_count - 1)

                    if j < len_np:
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

"""
Split by book so that every review of a book lands in the same set (train, validation or test).
Reviews of the same book share character names and plot details, so a random split would let the
model "recognize the book" instead of learning what a spoiler looks like. Splitting by book tests
the model on books it has never seen.

validation_size and testing_size are shares of the BOOKS, so the review counts come out close to,
but not exactly, those shares (books have different numbers of reviews).
"""
def split_dataset(df: pd.DataFrame, validation_size: float, testing_size: float, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout_size = validation_size + testing_size

    # First cut: (1 - holdout_size) of the books go to training, the rest are held out.
    # groups=df['book_id'] keeps all reviews of a book together.
    outer = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
    train_idx, holdout_idx = next(outer.split(df, groups=df['book_id']))
    training_data, holdout = df.iloc[train_idx], df.iloc[holdout_idx]

    # Second cut: split the held-out books into validation and test
    inner = GroupShuffleSplit(n_splits=1, test_size=testing_size / holdout_size, random_state=seed)
    val_idx, test_idx = next(inner.split(holdout, groups=holdout['book_id']))
    validation_data, testing_data = holdout.iloc[val_idx], holdout.iloc[test_idx]

    # Check that no book appears in more than one set
    train_books, val_books, test_books = (set(d['book_id']) for d in (training_data, validation_data, testing_data))
    assert not (train_books & val_books or train_books & test_books or val_books & test_books), "book appears in more than one split"

    return (training_data.reset_index(drop=True),
            validation_data.reset_index(drop=True),
            testing_data.reset_index(drop=True))

if __name__=="__main__":
    N = 10000
    PATH_TO_DATASET = "goodreads_reviews_spoiler_raw.json.gz"
    KEEP_COLS = ['review_id', 'book_id', 'review_text']
    SAVE_TO_CSV = "goodreads_reviews.csv"

    random.seed(100)

    # Create dataset and save as CSV
    # df = reservoir_sampling(PATH_TO_DATASET, N)
    df = stratified_sampling(PATH_TO_DATASET, N, 0.2, 0.8)
    df = clean_dataset(df, KEEP_COLS)
    df.to_csv(SAVE_TO_CSV)

    # Split the dataset into training, val, and testing by book (80/10/10 of the books)
    training_data, validation_data, testing_data = split_dataset(df, 0.1, 0.1)
    training_data.to_csv(f"training_{SAVE_TO_CSV}", index=False)
    validation_data.to_csv(f"validation_{SAVE_TO_CSV}", index=False)
    testing_data.to_csv(f"testing_{SAVE_TO_CSV}", index=False)

    # Output info
    print("Original Dataset:")
    get_dataset_info(df)
    print("Training Dataset:")
    get_dataset_info(training_data)
    print("Validation Dataset:")
    get_dataset_info(validation_data)
    print("Testing Dataset:")
    get_dataset_info(testing_data)


