"""
baseline.py — Scikit-Learn Naive Bayes baseline for comparison.

This module uses CountVectorizer + MultinomialNB from scikit-learn to provide
a reference implementation.  It is intentionally kept SEPARATE from the manual
implementation so the two never pollute each other.

Purpose:
    - Validate that the manual implementation produces comparable results.
    - Show students what a high-level library automates.
    - Provide timing comparison.

Usage:
    from backend.baseline import run_baseline
    results = run_baseline(X_train_raw, X_test_raw, y_train, y_test)
"""

import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def run_baseline(
    X_train_raw: list,
    X_test_raw: list,
    y_train: list,
    y_test: list,
    alpha: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Train and evaluate the sklearn CountVectorizer + MultinomialNB pipeline.

    Parameters
    ----------
    X_train_raw : list of str
        Raw (not preprocessed) training messages.
    X_test_raw : list of str
        Raw (not preprocessed) test messages.
    y_train : list of str
        Training labels ("spam" / "ham").
    y_test : list of str
        Test labels.
    alpha : float
        Laplace smoothing parameter (matched to manual implementation).
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict
        accuracy, precision_spam, recall_spam, f1_spam,
        train_time, predict_time, confusion_matrix
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("\n" + "=" * 60)
    log("  SCIKIT-LEARN BASELINE")
    log("=" * 60)

    # CountVectorizer handles its own tokenization.
    # Note: sklearn's default tokenizer is different from our manual one.
    # This is expected — the baseline shows what sklearn automates.
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"[a-zA-Z]+")
    classifier = MultinomialNB(alpha=alpha)

    # Train
    t0 = time.perf_counter()
    X_train_sk = vectorizer.fit_transform(X_train_raw)
    classifier.fit(X_train_sk, y_train)
    train_time = time.perf_counter() - t0

    # Predict
    t0 = time.perf_counter()
    X_test_sk = vectorizer.transform(X_test_raw)
    y_pred = classifier.predict(X_test_sk)
    predict_time = time.perf_counter() - t0

    y_test_arr = np.array(y_test)

    acc  = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, pos_label="spam")
    rec  = recall_score(y_test_arr,    y_pred, pos_label="spam")
    f1   = f1_score(y_test_arr,        y_pred, pos_label="spam")
    cm   = confusion_matrix(y_test_arr, y_pred, labels=["ham", "spam"])

    log(f"\n  Accuracy  : {acc:.4f}")
    log(f"  Precision : {prec:.4f}  (spam)")
    log(f"  Recall    : {rec:.4f}  (spam)")
    log(f"  F1-score  : {f1:.4f}  (spam)")
    log(f"  Train time: {train_time:.4f}s")
    log(f"  Pred time : {predict_time:.4f}s")

    log("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    log("              Ham    Spam")
    log(f"  Actual Ham  {cm[0,0]:>5}  {cm[0,1]:>5}")
    log(f"  Actual Spam {cm[1,0]:>5}  {cm[1,1]:>5}")

    log("\n  Full Classification Report:")
    log(classification_report(y_test_arr, y_pred, target_names=["ham", "spam"]))
    log("=" * 60)

    return {
        "accuracy": acc,
        "precision_spam": prec,
        "recall_spam": rec,
        "f1_spam": f1,
        "train_time": train_time,
        "predict_time": predict_time,
        "confusion_matrix": cm.tolist(),
        "vectorizer": vectorizer,
        "classifier": classifier,
    }
