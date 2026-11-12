"""
classifier.py — Manually implemented Multinomial Naive Bayes for spam/ham.

This module contains three logically distinct but related pieces:

1. build_vocabulary()       — vocabulary construction from training tokens
2. text_to_vector() /
   transform_documents()    — Bag-of-Words feature extraction
3. NaiveBayesClassifier     — full Multinomial Naive Bayes implementation

Everything here is implemented from scratch using NumPy.
No CountVectorizer, TfidfVectorizer, or MultinomialNB from scikit-learn.

=============================================================================
MATHEMATICAL REFERENCE
=============================================================================

Multinomial Naive Bayes
-----------------------
Given a document d with word-count vector x = [x_0, x_1, ..., x_{V-1}],
the log-score for class c is:

    score(c, d) = log P(c)  +  Σ_{i=0}^{V-1}  x_i * log P(word_i | c)

where:

    P(c) = N_c / N                          (class prior)

    P(word_i | c) = (count(word_i, c) + α)  (Laplace smoothed conditional)
                    ─────────────────────────
                    (total_words_c + α * V)

    N_c = number of training documents in class c
    N   = total number of training documents
    α   = smoothing parameter (default 1.0)
    V   = vocabulary size

The predicted class is:
    argmax_c  score(c, d)

Laplace smoothing (add-alpha smoothing) ensures P(word | c) > 0 even for
words that never appeared in training documents of class c.  Without it, a
single unseen word would zero-out the entire product P(d | c).

Log probabilities are used to avoid numerical underflow: multiplying many
small probabilities together quickly reaches floating-point underflow, but
adding logarithms does not.
"""

import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

Vocabulary = Dict[str, int]   # token → integer index


# ===========================================================================
# 1. Vocabulary Construction
# ===========================================================================

def build_vocabulary(token_lists: List[List[str]]) -> Vocabulary:
    """Build a token → index mapping from a collection of token lists.

    The vocabulary is created by scanning every token list in order.  Each
    *new* token receives the next available integer index starting from 0.
    Repeated tokens (across documents or within a document) do NOT receive
    new indices.

    IMPORTANT: Call this function only on training data.  Building the
    vocabulary from the full dataset before the train/test split would allow
    test-set information to leak into training (data leakage).

    Parameters
    ----------
    token_lists : list of list of str
        Preprocessed token lists for each training document.
        Example: [["free", "prize"], ["team", "meeting"]]

    Returns
    -------
    dict mapping str → int
        Vocabulary where each unique token has a unique integer index.
        Example: {"free": 0, "prize": 1, "team": 2, "meeting": 3}

    Examples
    --------
    >>> build_vocabulary([["free", "prize"], ["free", "money"]])
    {'free': 0, 'prize': 1, 'money': 2}
    >>> build_vocabulary([])
    {}
    """
    vocabulary: Vocabulary = {}
    for tokens in token_lists:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


# ===========================================================================
# 2. Bag-of-Words Feature Extraction
# ===========================================================================

def text_to_vector(tokens: List[str], vocabulary: Vocabulary) -> np.ndarray:
    """Convert a token list into a Bag-of-Words count vector.

    The output vector has length equal to the vocabulary size.  Position i
    holds the number of times the token mapped to index i appears in
    *tokens*.  Tokens not in *vocabulary* are silently ignored.

    Parameters
    ----------
    tokens : list of str
        Preprocessed tokens for one document.
    vocabulary : dict
        Token → integer-index mapping produced by :func:`build_vocabulary`.

    Returns
    -------
    np.ndarray, shape (V,), dtype int32
        Word-count vector.

    Examples
    --------
    >>> vocab = {"free": 0, "prize": 1, "meeting": 2}
    >>> text_to_vector(["free", "free", "prize"], vocab)
    array([2, 1, 0], dtype=int32)
    >>> text_to_vector(["unknown_word"], vocab)
    array([0, 0, 0], dtype=int32)
    >>> text_to_vector([], vocab)
    array([0, 0, 0], dtype=int32)
    """
    V = len(vocabulary)
    vector = np.zeros(V, dtype=np.int32)
    for token in tokens:
        idx = vocabulary.get(token)
        if idx is not None:
            vector[idx] += 1
    return vector


def transform_documents(
    token_lists: List[List[str]],
    vocabulary: Vocabulary,
) -> np.ndarray:
    """Convert a list of token lists into a Bag-of-Words matrix.

    Parameters
    ----------
    token_lists : list of list of str
        One token list per document.
    vocabulary : dict
        Token → index mapping.

    Returns
    -------
    np.ndarray, shape (n_documents, V), dtype int32
        Row i is the BoW vector for document i.

    Examples
    --------
    >>> vocab = {"free": 0, "prize": 1}
    >>> transform_documents([["free","free"], ["prize"]], vocab)
    array([[2, 0],
           [0, 1]], dtype=int32)
    """
    n = len(token_lists)
    V = len(vocabulary)
    X = np.zeros((n, V), dtype=np.int32)
    for i, tokens in enumerate(token_lists):
        X[i] = text_to_vector(tokens, vocabulary)
    return X


# ===========================================================================
# 3. Manual Multinomial Naive Bayes Classifier
# ===========================================================================

class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier implemented from scratch.

    Attributes set after :meth:`fit`
    --------------------------------
    classes_ : list of str
        Unique class labels, e.g. ["ham", "spam"].
    class_counts_ : dict {str: int}
        Number of training documents per class.
    word_counts_ : dict {str: np.ndarray}
        Raw word occurrence totals per class.  Shape (V,).
    total_words_ : dict {str: int}
        Total number of word occurrences per class (sum of word_counts_).
    class_log_prior_ : dict {str: float}
        log P(class) for each class.
    feature_log_prob_ : dict {str: np.ndarray}
        log P(word_i | class) for each class.  Shape (V,).
        Computed with Laplace smoothing.
    vocabulary_size_ : int
        V — vocabulary size seen during training.
    is_fitted_ : bool
        True once :meth:`fit` has completed successfully.

    Parameters
    ----------
    alpha : float, default 1.0
        Laplace (add-alpha) smoothing parameter.  Must be > 0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

        # Populated by fit()
        self.classes_: List[str] = []
        self.class_counts_: Dict[str, int] = {}
        self.word_counts_: Dict[str, np.ndarray] = {}
        self.total_words_: Dict[str, int] = {}
        self.class_log_prior_: Dict[str, float] = {}
        self.feature_log_prob_: Dict[str, np.ndarray] = {}
        self.vocabulary_size_: int = 0
        self.is_fitted_: bool = False

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y) -> "NaiveBayesClassifier":
        """Train the classifier.

        Computes class priors, word counts per class, and log-smoothed
        conditional probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, V)
            Bag-of-Words feature matrix (integer word counts).
        y : array-like of str, shape (n_samples,)
            Class labels for each sample ("spam" or "ham").

        Returns
        -------
        self
        """
        X = np.array(X, dtype=np.int64)
        y = np.array(y)

        n_samples, V = X.shape
        self.vocabulary_size_ = V
        self.classes_ = sorted(set(y))

        # ---- Class counts and priors --------------------------------------
        for c in self.classes_:
            mask = (y == c)
            self.class_counts_[c] = int(mask.sum())

            # log P(class) = log(N_c / N)
            self.class_log_prior_[c] = np.log(
                self.class_counts_[c] / n_samples
            )

        # ---- Word counts per class ----------------------------------------
        for c in self.classes_:
            mask = (y == c)
            # Sum each column over all documents of class c
            # word_counts_[c][i] = total occurrences of word i in class c
            self.word_counts_[c] = X[mask].sum(axis=0)          # shape (V,)
            self.total_words_[c] = int(self.word_counts_[c].sum())

        # ---- Laplace-smoothed log conditional probabilities ---------------
        # P(word_i | class) = (count(word_i, class) + alpha)
        #                     / (total_words_class + alpha * V)
        #
        # We compute log(P) and store it so that prediction is a cheap
        # dot product rather than a loop.
        for c in self.classes_:
            numerator = self.word_counts_[c] + self.alpha        # shape (V,)
            denominator = self.total_words_[c] + self.alpha * V  # scalar
            # log probability for each word given this class
            self.feature_log_prob_[c] = np.log(numerator / denominator)

        self.is_fitted_ = True
        return self

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model has not been trained.  Call fit() first."
            )

    def predict_log_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-score matrix for all classes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, V)
            BoW feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Column order matches ``self.classes_``.

        Score formula
        -------------
            score(c, d) = log P(c)  +  x · log P(word | c)

        The dot product x · log P(word | c) is the Multinomial NB formulation
        where each word's log-probability is weighted by how many times it
        appears in the document.
        """
        self._check_fitted()
        X = np.array(X, dtype=np.int64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for j, c in enumerate(self.classes_):
            # log P(class) is the same for all documents
            log_prior = self.class_log_prior_[c]
            # feature_log_prob_[c] has shape (V,)
            # X has shape (n, V)
            # X @ feature_log_prob_[c] gives shape (n,) — one score per doc
            log_likelihood = X @ self.feature_log_prob_[c]
            scores[:, j] = log_prior + log_likelihood

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class for each sample in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, V)
            BoW feature matrix.

        Returns
        -------
        np.ndarray of str, shape (n_samples,)
            Predicted class labels.
        """
        scores = self.predict_log_scores(X)
        # argmax along class axis → index of highest-scoring class
        best_class_idx = np.argmax(scores, axis=1)
        classes_arr = np.array(self.classes_)
        return classes_arr[best_class_idx]

    def predict_one(self, x: np.ndarray) -> str:
        """Predict the class for a single BoW vector.

        Parameters
        ----------
        x : np.ndarray, shape (V,)
            BoW feature vector for one document.

        Returns
        -------
        str
            Predicted class label ("spam" or "ham").
        """
        self._check_fitted()
        x = np.array(x, dtype=np.int64).reshape(1, -1)
        return str(self.predict(x)[0])

    def score(self, X: np.ndarray, y) -> float:
        """Return classification accuracy on the given data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, V)
        y : array-like of str, shape (n_samples,)

        Returns
        -------
        float
            Fraction of correctly classified samples (0.0 – 1.0).
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.array(y)))

    def get_scores_for_text(
        self,
        tokens: List[str],
        vocabulary: Vocabulary,
    ) -> Dict[str, float]:
        """Convenience: return the raw log score for each class given tokens.

        Useful for the API and the frontend score display.

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens for the input text.
        vocabulary : dict
            The same vocabulary used during training.

        Returns
        -------
        dict {class_label: float}
            Raw log scores (NOT calibrated probabilities).
        """
        self._check_fitted()
        x = text_to_vector(tokens, vocabulary).reshape(1, -1)
        scores = self.predict_log_scores(x)[0]
        return {c: float(s) for c, s in zip(self.classes_, scores)}

    # -----------------------------------------------------------------------
    # Introspection helpers (for learning / debugging)
    # -----------------------------------------------------------------------

    def top_words(self, class_label: str, n: int = 20) -> List[tuple]:
        """Return the top-n words with highest log P(word | class).

        This is useful for understanding which words the classifier
        considers most indicative of spam or ham.

        Parameters
        ----------
        class_label : str
            "spam" or "ham"
        n : int
            Number of top words to return.

        Returns
        -------
        list of (index, log_prob) tuples, sorted descending by log_prob.
        """
        self._check_fitted()
        if class_label not in self.classes_:
            raise ValueError(f"Unknown class: {class_label!r}")
        log_probs = self.feature_log_prob_[class_label]
        top_indices = np.argsort(log_probs)[::-1][:n]
        return [(int(i), float(log_probs[i])) for i in top_indices]

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"NaiveBayesClassifier("
            f"alpha={self.alpha}, "
            f"classes={self.classes_}, "
            f"status={status})"
        )
