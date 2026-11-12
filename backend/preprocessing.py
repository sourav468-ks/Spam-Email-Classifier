"""
preprocessing.py — Text preprocessing pipeline for the Spam Email Classifier.

Responsibilities (single):
    Convert raw email text into a clean list of tokens.

Pipeline
--------
    Raw text
       ↓ unicode normalization (NFKD → ASCII)
       ↓ lowercase
       ↓ remove non-alphabetic characters (keep spaces)
       ↓ normalize whitespace
       ↓ split on whitespace → token list
       ↓ filter empty tokens

IMPORTANT: This pipeline must be applied identically during training and
during prediction.  Using a different cleaning method at prediction time
degrades model accuracy in hard-to-debug ways.

Design choices
--------------
* We keep **only alphabetic characters**.  Digits, currency symbols, and
  punctuation are stripped.  This is intentionally simple — the goal is to
  understand the Naive Bayes mathematics, not to build a production-grade
  email security product.
* We do NOT remove stop words by default.  Stop words carry some signal for
  spam (e.g. "free", "now", "click") and their removal is an optional
  optimization added later.
* We do NOT stem tokens.  Stemming adds complexity without helping
  understand the core algorithm.
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize raw text into a clean, lowercase, letters-only string.

    Steps:
        1. Coerce to str (guard against None / non-string input).
        2. Unicode NFKD normalization + encode to ASCII (drops accents).
        3. Lowercase.
        4. Replace non-alphabetic characters with a space.
        5. Collapse multiple whitespace characters into a single space.
        6. Strip leading / trailing whitespace.

    Parameters
    ----------
    text : str
        Raw email text.

    Returns
    -------
    str
        Clean, lowercase, letters-and-spaces-only string.

    Examples
    --------
    >>> clean_text("FREE!!! WIN $$$")
    'free win'
    >>> clean_text("Héllo Wörld")
    'hello world'
    >>> clean_text("")
    ''
    >>> clean_text("   ")
    ''
    """
    # Guard: coerce to string
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Unicode normalization → ASCII (strips accented characters)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # 2. Lowercase
    text = text.lower()

    # 3. Replace every non-alphabetic character with a space
    #    (digits, punctuation, symbols → space)
    text = re.sub(r"[^a-z]", " ", text)

    # 4. Collapse runs of whitespace to a single space and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list:
    """Split a clean string into a list of tokens.

    Assumes *text* has already been cleaned by :func:`clean_text`.
    Splitting on whitespace is sufficient after cleaning.

    Parameters
    ----------
    text : str
        Clean text (output of :func:`clean_text`).

    Returns
    -------
    list of str
        List of token strings.  Empty string input produces an empty list.

    Examples
    --------
    >>> tokenize("free win now")
    ['free', 'win', 'now']
    >>> tokenize("")
    []
    """
    if not text:
        return []
    return text.split()


def preprocess(text: str) -> list:
    """Full preprocessing pipeline: clean + tokenize.

    This is the function that should be called everywhere — during training,
    vocabulary construction, feature extraction, and prediction.

    Parameters
    ----------
    text : str
        Raw email text (may contain any characters).

    Returns
    -------
    list of str
        Ordered list of lowercase alphabetic tokens.
        Empty or whitespace-only input returns an empty list.

    Examples
    --------
    >>> preprocess("FREE!!! WIN $$$")
    ['free', 'win']
    >>> preprocess("Congratulations! You WON a free prize!!!")
    ['congratulations', 'you', 'won', 'a', 'free', 'prize']
    >>> preprocess("")
    []
    >>> preprocess(None)
    []
    """
    return tokenize(clean_text(text))


def preprocess_batch(texts) -> list:
    """Apply :func:`preprocess` to an iterable of texts.

    Parameters
    ----------
    texts : iterable of str
        Collection of raw email strings.

    Returns
    -------
    list of list of str
        Each inner list is the token list for one email.
    """
    return [preprocess(t) for t in texts]
