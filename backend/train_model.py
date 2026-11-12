"""
train_model.py — End-to-end training pipeline for the Spam Email Classifier.

Pipeline
--------
    load_dataset()
         ↓
    train_test_split()          ← sklearn utility (allowed)
         ↓
    preprocess_batch()          ← training messages only
         ↓
    build_vocabulary()          ← training tokens only (no data leakage)
         ↓
    transform_documents()       ← training set
         ↓
    model.fit()                 ← manual Naive Bayes
         ↓
    transform_documents()       ← test set (same vocabulary, read-only)
         ↓
    model.predict()
         ↓
    evaluate()                  ← sklearn metrics (allowed)
         ↓
    save_model()                ← joblib

Usage
-----
    python backend/train_model.py

Or from another module:
    from backend.train_model import train_and_evaluate
"""

import os
import sys
import time

# Allow running from project root or from backend/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.config import (
    DATASET_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    ALPHA,
    MODEL_PATH,
    VOCABULARY_PATH,
    MODEL_DIR,
)
from backend.dataset_loader import load_dataset, describe_dataset
from backend.preprocessing import preprocess_batch
from backend.classifier import build_vocabulary, transform_documents, NaiveBayesClassifier


# ===========================================================================
# Main training function
# ===========================================================================

def train_and_evaluate(
    dataset_path: str = DATASET_PATH,
    alpha: float = ALPHA,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full training pipeline and return a result dictionary.

    Parameters
    ----------
    dataset_path : str
        Path to spam.csv.
    alpha : float
        Laplace smoothing parameter.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Fixed seed for reproducibility.
    save : bool
        Whether to persist the model and vocabulary to disk.
    verbose : bool
        Whether to print progress to stdout.

    Returns
    -------
    dict with keys:
        model, vocabulary, X_train, X_test, y_train, y_test,
        y_pred, metrics, train_time, predict_time
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------ 1. Load
    log("\n" + "=" * 60)
    log("  SPAM EMAIL CLASSIFIER - TRAINING PIPELINE")
    log("=" * 60)

    df = load_dataset(dataset_path)
    if verbose:
        describe_dataset(df)

    messages = df["message"].tolist()
    labels = df["label"].tolist()

    # ------------------------------------------------------------------ 2. Split
    # Split BEFORE preprocessing or vocabulary construction.
    # This is the critical guard against data leakage.
    log("\n[train_model] Splitting dataset ...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    log(f"  Training samples : {len(X_train_raw)}")
    log(f"  Testing  samples : {len(X_test_raw)}")

    # ------------------------------------------------------------------ 3. Preprocess
    log("\n[train_model] Preprocessing text ...")
    train_tokens = preprocess_batch(X_train_raw)   # list of list of str
    test_tokens  = preprocess_batch(X_test_raw)

    # ------------------------------------------------------------------ 4. Vocabulary
    # Built ONLY from training tokens
    log("\n[train_model] Building vocabulary ...")
    vocabulary = build_vocabulary(train_tokens)
    log(f"  Vocabulary size  : {len(vocabulary)}")

    # ------------------------------------------------------------------ 5. Vectorize
    log("\n[train_model] Vectorizing documents ...")
    X_train = transform_documents(train_tokens, vocabulary)
    X_test  = transform_documents(test_tokens,  vocabulary)
    log(f"  X_train shape    : {X_train.shape}")
    log(f"  X_test  shape    : {X_test.shape}")

    # ------------------------------------------------------------------ 6. Train
    log("\n[train_model] Training Manual Naive Bayes ...")
    model = NaiveBayesClassifier(alpha=alpha)
    t0 = time.perf_counter()
    model.fit(X_train, np.array(y_train))
    train_time = time.perf_counter() - t0
    log(f"  Training time    : {train_time:.4f}s")

    # ------------------------------------------------------------------ 7. Predict
    log("\n[train_model] Generating predictions ...")
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0
    log(f"  Prediction time  : {predict_time:.4f}s")

    # ------------------------------------------------------------------ 8. Evaluate
    log("\n" + "=" * 60)
    log("  EVALUATION RESULTS")
    log("=" * 60)

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

    log("\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    log("              Ham    Spam")
    log(f"  Actual Ham  {cm[0,0]:>5}  {cm[0,1]:>5}")
    log(f"  Actual Spam {cm[1,0]:>5}  {cm[1,1]:>5}")

    log("\n  Full Classification Report:")
    log(classification_report(y_test_arr, y_pred, target_names=["ham", "spam"]))

    metrics = {
        "accuracy": acc,
        "precision_spam": prec,
        "recall_spam": rec,
        "f1_spam": f1,
        "confusion_matrix": cm.tolist(),
        "train_time": train_time,
        "predict_time": predict_time,
        "vocab_size": len(vocabulary),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }

    # ------------------------------------------------------------------ 9. Save
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vocabulary, VOCABULARY_PATH)
        log(f"\n[train_model] Model saved      -> {MODEL_PATH}")
        log(f"[train_model] Vocabulary saved -> {VOCABULARY_PATH}")

    log("\n" + "=" * 60)
    log("  TRAINING COMPLETE")
    log("=" * 60 + "\n")

    return {
        "model": model,
        "vocabulary": vocabulary,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "train_time": train_time,
        "predict_time": predict_time,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
    }


# ===========================================================================
# Quick sanity check - predict a few hand-crafted examples
# ===========================================================================

def quick_demo(model: NaiveBayesClassifier, vocabulary: dict) -> None:
    """Print predictions for a few obvious spam/ham examples."""
    from backend.preprocessing import preprocess
    from backend.classifier import text_to_vector

    examples = [
        ("Congratulations! You won a FREE prize. Claim now!!!", "spam"),
        ("Can we meet tomorrow at 10 AM for the project review?", "ham"),
        ("URGENT: Your bank account has been suspended. Click here.", "spam"),
        ("Hi, are you coming to the team lunch on Friday?", "ham"),
        ("Win a $1,000,000 lottery - limited time offer!", "spam"),
    ]

    print("\n  QUICK DEMO - Sanity Check")
    print("  " + "-" * 58)
    print(f"  {'Expected':<8} {'Predicted':<10}  Message (truncated)")
    print("  " + "-" * 58)
    for text, expected in examples:
        tokens = preprocess(text)
        x = text_to_vector(tokens, vocabulary).reshape(1, -1)
        pred = model.predict(x)[0]
        scores = model.get_scores_for_text(tokens, vocabulary)
        match = "[OK]" if pred == expected else "[FAIL]"
        print(f"  {expected:<8} {pred:<10} {match:<6}  {text[:50]}")

    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    from backend.baseline import run_baseline

    results = train_and_evaluate(verbose=True, save=True)
    quick_demo(results["model"], results["vocabulary"])

    # Run Scikit-Learn baseline comparison
    run_baseline(
        X_train_raw=results["X_train_raw"],
        X_test_raw=results["X_test_raw"],
        y_train=results["y_train"],
        y_test=results["y_test"],
        alpha=ALPHA,
        verbose=True,
    )
