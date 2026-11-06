"""
Evaluation helpers shared by every model.

Use from another script:
    from evaluation import evaluate, top_weighted_terms

Or run on its own after training, from the repo root:
    python models/evaluation.py

Running it reads the files that train_logreg.py saves in models/output/:
    {split}_predictions.csv  - one row per review with the true label and each model's prediction
    logreg.joblib            - the fitted TF-IDF vectorizer and logistic regression model
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

OUT = Path(__file__).resolve().parent / "output"
LABEL_COLUMN = "has_spoiler"


def safe_divide(numerator, denominator) -> float:
    # e.g. the baseline never predicts a spoiler, so TP + FP = 0 and precision would be 0/0
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate(y_true, y_pred) -> dict:
    """
    Score any model's predictions. Spoiler (1) is the positive class.

    confusion_matrix with labels=[0, 1] returns [[TN, FP], [FN, TP]]:
    rows are the true label, columns the prediction.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Of the reviews we flagged as spoilers, how many really were
    precision = safe_divide(tp, tp + fp)
    # Of the real spoilers, how many we caught
    recall = safe_divide(tp, tp + fn)
    # Harmonic mean of precision and recall
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


# Small examples to check evaluate()
_perfect = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
assert np.isclose(_perfect["f1"], 1) and np.isclose(_perfect["accuracy"], 1)
_never_spoiler = evaluate([0, 0, 1, 1], [0, 0, 0, 0])
assert _never_spoiler["precision"] == 0 and _never_spoiler["recall"] == 0 and _never_spoiler["f1"] == 0
_half = evaluate([0, 0, 1, 1], [0, 1, 1, 0])  # TP=1, FP=1, FN=1
assert np.isclose(_half["precision"], 0.5) and np.isclose(_half["recall"], 0.5) and np.isclose(_half["f1"], 0.5)
_mixed = evaluate([1, 1, 1, 0], [1, 1, 0, 1])  # TP=2, FP=1, FN=1 -> P=R=2/3
assert np.isclose(_mixed["f1"], 2 / 3)
assert _mixed["confusion_matrix"] == [[0, 1], [1, 2]]


def top_weighted_terms(model, vectorizer, n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Words with the strongest weights in a linear model (e.g. logistic regression).
    Positive weights push toward spoiler, negative toward non-spoiler.
    """
    terms = pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "weight": model.coef_[0],
    })
    spoiler = terms.nlargest(n, "weight").reset_index(drop=True)
    non_spoiler = terms.nsmallest(n, "weight").reset_index(drop=True)
    return spoiler, non_spoiler


def format_scores(name: str, s: dict) -> str:
    return (f"{name:10s} acc={s['accuracy']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} "
            f"F1={s['f1']:.3f} CM={s['confusion_matrix']}")


def main():
    for split in ("validation", "test"):
        preds = pd.read_csv(OUT / f"{split}_predictions.csv")
        print(f"== {split}")
        for col in [c for c in preds.columns if c.startswith("pred_")]:
            print(format_scores(col.removeprefix("pred_"), evaluate(preds[LABEL_COLUMN], preds[col])))

    saved = joblib.load(OUT / "logreg.joblib")
    spoiler, non_spoiler = top_weighted_terms(saved["model"], saved["vectorizer"])
    print("\nTop spoiler terms:\n", spoiler.round(3).to_string(index=False))
    print("\nTop non-spoiler terms:\n", non_spoiler.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
