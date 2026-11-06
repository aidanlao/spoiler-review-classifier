"""
Evaluation helpers shared by every model.

Use from another script:
    from evaluation import evaluate, top_weighted_terms

Or run on its own after training, from the repo root:
    python models/evaluation.py

Running it reads the files that train_logreg.py saves in models/output/:
    {split}_predictions.csv  - one row per review with the true label and each model's prediction
    logreg.joblib            - the fitted TF-IDF vectorizer and logistic regression model
and saves:
    confusion_matrices_{split}.png - one confusion matrix per model, side by side
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
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


def plot_confusion_matrices(y_true, predictions: dict, title: str, save_path: Path):
    """
    One confusion matrix per model, side by side (same style as Assignment 2).

    predictions maps a model name to its 0/1 predictions, e.g.
        {"Baseline": preds["pred_baseline"], "Logistic regression": preds["pred_logreg"]}
    Rows are the true label, columns the prediction.
    """
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.5))
    axes = np.atleast_1d(axes)  # so a single model still gives a list of axes

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        s = evaluate(y_true, y_pred)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        # Write each count in its cell; white text on dark cells, black on light ones
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=14)
        ax.set(title=f"{name}\nP={s['precision']:.2f}  R={s['recall']:.2f}  F1={s['f1']:.2f}",
               xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], yticks=[0, 1],
               xticklabels=["No spoiler", "Spoiler"],
               yticklabels=["No spoiler", "Spoiler"])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# Readable names for the prediction columns saved by train_logreg.py
MODEL_NAMES = {"pred_baseline": "Majority baseline", "pred_logreg": "Logistic regression"}


def main():
    for split in ("validation", "test"):
        preds = pd.read_csv(OUT / f"{split}_predictions.csv")
        pred_cols = [c for c in preds.columns if c.startswith("pred_")]
        print(f"== {split}")
        for col in pred_cols:
            print(format_scores(col.removeprefix("pred_"), evaluate(preds[LABEL_COLUMN], preds[col])))

        save_path = OUT / f"confusion_matrices_{split}.png"
        plot_confusion_matrices(
            preds[LABEL_COLUMN],
            {MODEL_NAMES.get(c, c): preds[c] for c in pred_cols},
            title=f"Confusion matrices ({split} set)",
            save_path=save_path,
        )
        print(f"Saved {save_path.name}")

    saved = joblib.load(OUT / "logreg.joblib")
    spoiler, non_spoiler = top_weighted_terms(saved["model"], saved["vectorizer"])
    print("\nTop spoiler terms:\n", spoiler.round(3).to_string(index=False))
    print("\nTop non-spoiler terms:\n", non_spoiler.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
