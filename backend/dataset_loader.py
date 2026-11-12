"""
dataset_loader.py — Responsible for loading and validating the spam/ham dataset.

Responsibilities (single):
    Load a CSV, validate its structure, clean invalid rows, normalize labels,
    and return a clean DataFrame.

Nothing else lives here — no ML, no preprocessing, no I/O beyond reading the
one dataset file.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the spam/ham CSV dataset and return a clean DataFrame.

    The CSV must have at least two columns: ``label`` and ``message``.
    Labels are normalized to lowercase ``"spam"`` / ``"ham"``.
    Rows with missing or empty messages, and rows with unrecognized labels,
    are dropped.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: ``label`` (str), ``message`` (str)

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    ValueError
        If the CSV is missing required columns or has no usable rows after
        cleaning.
    """
    # ---- 1. File existence check ----------------------------------------
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Run `python dataset/download_dataset.py` to fetch it."
        )

    # ---- 2. Load CSV -------------------------------------------------------
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {path!r}: {exc}") from exc

    # ---- 3. Column validation ----------------------------------------------
    required = {"label", "message"}
    missing_cols = required - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(
            f"Dataset must contain 'label' and 'message' columns. "
            f"Missing: {missing_cols}. Found: {list(df.columns)}"
        )

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower()
    df = df[["label", "message"]].copy()

    # ---- 4. Drop invalid rows ---------------------------------------------
    original_count = len(df)

    # Drop rows where either column is NaN
    df.dropna(subset=["label", "message"], inplace=True)

    # Convert to string (guards against accidental numeric columns)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    # Drop empty messages
    df = df[df["message"].str.len() > 0]

    # ---- 5. Normalize labels -----------------------------------------------
    # Map common alternative representations
    label_map = {
        "spam": "spam",
        "ham": "ham",
        "1": "spam",
        "0": "ham",
        "junk": "spam",
        "legit": "ham",
    }
    df["label"] = df["label"].map(label_map)

    # Drop rows whose label did not match any known value
    df.dropna(subset=["label"], inplace=True)

    dropped = original_count - len(df)
    if dropped > 0:
        print(f"[dataset_loader] Dropped {dropped} invalid row(s).")

    # ---- 6. Final check ----------------------------------------------------
    if len(df) == 0:
        raise ValueError(
            "No usable rows remain after cleaning. "
            "Check the dataset format and label values."
        )

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def describe_dataset(df: pd.DataFrame) -> None:
    """Print a short summary of the dataset to stdout."""
    total = len(df)
    counts = df["label"].value_counts()
    print(f"[dataset_loader] Loaded {total} rows.")
    for label, count in counts.items():
        pct = 100.0 * count / total
        print(f"  {label:<6}: {count:>5} ({pct:.1f}%)")
