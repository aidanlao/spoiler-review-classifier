import pandas as pd
import numpy as np

def get_dataset_info(df: pd.DataFrame):
    print("-------------------------------------------------------------")
    # Number of reviews:
    print(f"Number of reviews: {len(df)}")
    print()

    # Number of spoilers and non spoilers:
    num_spoiler = len(df.loc[df['has_spoiler'] == 1])
    num_non_spoiler = len(df.loc[df['has_spoiler'] == 0])
    print(f"Number of spoilers: {num_spoiler}")
    print(f"Number of non-spoilers: {num_non_spoiler}")
    print(f"Reviews consist of {(num_spoiler/len(df)):.2%} spoilers and {(num_non_spoiler/len(df)):.2%} non-spoilers")
    print()

    # Number of unique books
    print(f"Number of unique books: {df['book_id'].nunique()}")
    print("Top 5 books with the most reviews:")
    print(f"{df.groupby('book_id').size().nlargest(5).reset_index(name='# of reviews')}")
    print("-------------------------------------------------------------")
    print()