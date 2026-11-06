"""
TF-IDF features, majority-class baseline and logistic regression.

Run from the repo root:
    python models/train_logreg.py

Reads the book-based splits made by dataset/create_dataset.py:
    dataset/{training,validation,testing}_goodreads_reviews.csv
and writes to models/output/:
    {validation,test}_predictions.csv    - each review's label and each model's prediction
    logreg.joblib                        - fitted vectorizer + final logistic regression
    results.json                         - split sizes, tuning grid and scores
Scoring is done with evaluation.py.

Overall the script does 7 things in order:
    1. load the three splits
    2. check that no book appears in more than one split
    3. turn the review text into numbers (TF-IDF)
    4. train the "always no spoiler" baseline
    5. tune logistic regression on the validation set
    6. save predictions, the model and the results
    7. print a summary
"""
# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import json                 # to save results.json
from pathlib import Path    # to build file paths that work on any computer (Mac/Windows/Linux)

import joblib               # to save the trained model to a file (comes with scikit-learn)
import pandas as pd         # tables of data ("DataFrames"); df = one table, rows = reviews, columns = fields
from sklearn.dummy import DummyClassifier                    # the "always guess the most common class" baseline
from sklearn.feature_extraction.text import TfidfVectorizer  # turns review text into numbers
from sklearn.linear_model import LogisticRegression          # our model

# Our own scoring functions from evaluation.py (same folder)
from evaluation import evaluate, format_scores, top_weighted_terms

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
# Fixes all randomness so every run gives the same split and results
# (same convention as the tutorials and assignments).
SEED = 42

# File locations. ROOT is the repo folder, found from where THIS file lives
# (models/train_logreg.py -> go up two levels), so the script works no matter
# which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"                          # where create_dataset.py saves the splits
SPLIT_FILES = {                                     # our name for each split -> file name
    "train": "training_goodreads_reviews.csv",
    "validation": "validation_goodreads_reviews.csv",
    "test": "testing_goodreads_reviews.csv",
}
OUT = ROOT / "models" / "output"                    # where everything this script makes is saved

# TF-IDF settings (see the Features section of the report)
TFIDF_PARAMS = dict(
    lowercase=True,       # "Ending" and "ending" count as the same word
    ngram_range=(1, 2),   # use single words AND two-word phrases, e.g. "she dies", "the end"
    min_df=2,             # ignore terms that appear in only one review (typos, one-off names)
    max_df=0.9,           # ignore terms that appear in more than 90% of reviews (too common to help)
    sublinear_tf=True,    # use 1 + log(count): a word repeated 10 times counts ~3.3x, not 10x,
                          # so long rants that repeat words don't dominate
    max_features=50000,   # keep at most the 50,000 most common terms
)

# Values tried during tuning. 5 C values x 2 class-weight options = 10 models.
# C is the INVERSE of regularization strength (C = 1/alpha in lecture notation):
#   small C = strong regularization = small weights, simpler model (risk: underfitting)
#   large C = weak regularization   = weights can grow, fits training data closely (risk: overfitting)
C_VALUES = [0.01, 0.1, 1, 10, 100]
# None       = every review counts equally in the loss
# "balanced" = weight each class inversely to how common it is, so a missed spoiler
#              (the rare class, ~20%) costs about 4x as much as a false alarm
CLASS_WEIGHTS = [None, "balanced"]


# ---------------------------------------------------------------------------
# LOADING THE SPLITS
# ---------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    """
    Load one of the book-based splits made by dataset/create_dataset.py.
    Each split is a DataFrame (table): one row per review, one column per field.
    """
    d = pd.read_csv(DATASET / SPLIT_FILES[name])
    # Replace any empty review text with "" and make sure everything is a string,
    # so the vectorizer doesn't crash on a missing value
    d["review_text"] = d["review_text"].fillna("").astype(str)
    return d


def describe(name: str, d: pd.DataFrame) -> dict:
    """Summary of one split. Used for the split table in the report."""
    return {
        "split": name,
        "reviews": len(d),                               # number of rows
        "books": int(d["book_id"].nunique()),            # number of different books
        "spoilers": int(d["has_spoiler"].sum()),         # labels are 0/1, so the sum = number of spoilers
        "spoiler_share": float(d["has_spoiler"].mean()), # mean of 0/1 labels = fraction that are spoilers
    }


def main():
    # -----------------------------------------------------------------------
    # 1. LOAD THE THREE SPLITS
    # -----------------------------------------------------------------------
    # Create models/output/ if it doesn't exist (no error if it already does)
    OUT.mkdir(parents=True, exist_ok=True)

    # The split itself is done by book in dataset/create_dataset.py (see split_dataset there)
    train, val, test = load_split("train"), load_split("validation"), load_split("test")

    # -----------------------------------------------------------------------
    # 2. CHECK THE SPLIT IS BY BOOK
    # -----------------------------------------------------------------------
    # Safety check in case the CSVs were made some other way: collect the set of book IDs
    # in each split. "&" is set intersection, so if any two splits share a book, stop with an error.
    books = [set(d["book_id"]) for d in (train, val, test)]
    assert not (books[0] & books[1] or books[0] & books[2] or books[1] & books[2]), \
        "a book appears in more than one split - rerun dataset/create_dataset.py"

    # One summary dictionary per split (reviews, books, spoilers, spoiler share)
    split_info = [describe(n, d) for n, d in (("train", train), ("validation", val), ("test", test))]

    # -----------------------------------------------------------------------
    # 3. TEXT -> NUMBERS (TF-IDF)
    # -----------------------------------------------------------------------
    # Each review becomes a row of 50,000 numbers (one per word/phrase in the vocabulary).
    # Most are 0 because a review only uses a few hundred words, so it's stored as a
    # "sparse" matrix that only keeps the non-zero values.
    # A word's number is high if it's frequent in THIS review but rare across all reviews.
    vec = TfidfVectorizer(**TFIDF_PARAMS)  # ** unpacks the settings dictionary into arguments

    # fit_transform on TRAINING ONLY: learns the vocabulary and how rare each word is,
    # then converts the training reviews.
    X_train = vec.fit_transform(train["review_text"])
    # transform (no fit) for validation/test: uses the training vocabulary, learns nothing new.
    # Same idea as scaling with training statistics in Tutorial 3. Words that only appear
    # in validation/test books are ignored, like they would be for a brand-new book.
    X_val = vec.transform(val["review_text"])
    X_test = vec.transform(test["review_text"])

    # The answers (0 = no spoiler, 1 = spoiler) as arrays.
    # y_test isn't needed here because the test set is scored from the saved predictions below.
    y_train, y_val = train["has_spoiler"].values, val["has_spoiler"].values

    # -----------------------------------------------------------------------
    # 4. MAJORITY-CLASS BASELINE
    # -----------------------------------------------------------------------
    # Looks at the training labels, sees "no spoiler" is most common, and always predicts 0.
    # Any real model has to beat this on spoiler-class F1 (its F1 is 0 because it never
    # catches a spoiler, even though its accuracy is ~80%).
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # 5. TUNE LOGISTIC REGRESSION ON THE VALIDATION SET
    # -----------------------------------------------------------------------
    # Logistic regression: p(spoiler | review) = sigmoid(w . x + b), trained by minimizing
    # cross-entropy (log) loss + an L2 penalty on the weights (regularization).
    grid = []  # one row per setting tried
    for cw in CLASS_WEIGHTS:        # None, then "balanced"
        for c in C_VALUES:          # 0.01, 0.1, 1, 10, 100
            # solver="liblinear": the optimization algorithm (coordinate descent, works well
            #   for sparse text features); max_iter=1000: cap on how long it can run
            model = LogisticRegression(C=c, class_weight=cw, solver="liblinear", max_iter=1000, random_state=SEED)
            model.fit(X_train, y_train)                # learn the weights on training data
            s = evaluate(y_val, model.predict(X_val))  # predict on VALIDATION and score it
            # Record this setting and its scores. `cw or "none"` turns None into readable text.
            # **{...} copies the 4 score values into the same dictionary.
            grid.append({"C": c, "class_weight": cw or "none", **{k: s[k] for k in ("accuracy", "precision", "recall", "f1")}})

    # Pick the setting with the highest validation F1 (spoiler class).
    # The test set is NOT used for this choice.
    best = max(grid, key=lambda g: g["f1"])
    # Turn the text "none" back into Python's None so scikit-learn understands it
    best_cw = None if best["class_weight"] == "none" else best["class_weight"]

    # Train the final model with the winning settings (on the training set).
    # Note: Tutorial 3 instead refits on training + validation before testing; that would change here.
    final = LogisticRegression(C=best["C"], class_weight=best_cw, solver="liblinear", max_iter=1000, random_state=SEED)
    final.fit(X_train, y_train)
    # Save the vectorizer and model together so evaluation.py can reload them without retraining
    joblib.dump({"vectorizer": vec, "model": final}, OUT / "logreg.joblib")

    # -----------------------------------------------------------------------
    # 6. SAVE PREDICTIONS AND RESULTS
    # -----------------------------------------------------------------------
    # Save each review's label and predictions so evaluation.py can score them without retraining
    results = {
        "splits": split_info,
        "vocab_size": len(vec.vocabulary_),
        # JSON can't store tuples, so (1, 2) is converted to [1, 2]
        "tfidf": {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_PARAMS.items()},
        "grid": grid,
        "best": {"C": best["C"], "class_weight": best["class_weight"]},
    }
    for name, d, X in (("validation", val, X_val), ("test", test, X_test)):
        preds = d.copy()                                # copy the reviews table for this split
        preds["pred_baseline"] = baseline.predict(X)    # baseline's 0/1 prediction (always 0)
        preds["pred_logreg"] = final.predict(X)         # logistic regression's 0/1 prediction
        # predict_proba gives [P(no spoiler), P(spoiler)] for each review; [:, 1] keeps P(spoiler).
        # Useful for finding confident mistakes (e.g. P(spoiler) = 0.95 but it wasn't a spoiler).
        # A review is predicted as a spoiler when this is >= 0.5.
        preds["prob_logreg"] = final.predict_proba(X)[:, 1]
        preds.to_csv(OUT / f"{name}_predictions.csv", index=False)
        # Score both models with our evaluate() from evaluation.py
        results[name] = {
            "baseline": evaluate(preds["has_spoiler"], preds["pred_baseline"]),
            "logreg": evaluate(preds["has_spoiler"], preds["pred_logreg"]),
        }

    # The 20 words with the largest positive weights (push toward spoiler)
    # and largest negative weights (push toward non-spoiler)
    spoiler_terms, non_spoiler_terms = top_weighted_terms(final, vec)
    results["top_spoiler_terms"] = spoiler_terms["term"].tolist()
    results["top_non_spoiler_terms"] = non_spoiler_terms["term"].tolist()

    # Write everything to results.json (indent=2 makes it readable)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------------
    # 7. PRINT A SUMMARY
    # -----------------------------------------------------------------------
    print(pd.DataFrame(split_info).to_string(index=False))           # split table
    print(f"\nVocabulary size: {results['vocab_size']}")
    print("\nValidation grid:")
    print(pd.DataFrame(grid).round(3).to_string(index=False))        # tuning table
    print(f"\nBest: C={best['C']}, class_weight={best['class_weight']}")
    for split in ("validation", "test"):                             # final scores
        print(f"== {split}")
        for m in ("baseline", "logreg"):
            print(format_scores(m, results[split][m]))


# Only run main() when this file is run directly (python models/train_logreg.py).
# If another file imports this one (e.g. to reuse load_split), nothing trains automatically.
if __name__ == "__main__":
    main()
