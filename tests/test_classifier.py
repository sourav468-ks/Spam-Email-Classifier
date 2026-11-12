"""
tests/test_classifier.py — Unit tests for the Naive Bayes classifier.

Tests cover:
    NaiveBayesClassifier.__init__()
        - Default alpha
        - Invalid alpha raises ValueError

    NaiveBayesClassifier.fit()
        - Correct class counts
        - Correct word counts
        - Laplace smoothing (zero-count words have P > 0)
        - Log priors sum correctly
        - Feature log prob shape

    NaiveBayesClassifier.predict()
        - Returns known class labels
        - Single sample prediction
        - Batch prediction shape

    NaiveBayesClassifier.predict_one()
        - Returns string label

    NaiveBayesClassifier.score()
        - Perfect accuracy on training data (tiny example)

    predict_log_scores()
        - Shape correctness
        - Highest score matches prediction

    Untrained model
        - predict() raises RuntimeError

    Mathematical validation (TRD §27)
        - 4-document toy example with manually computed expected values
        - Word counts
        - Laplace-smoothed conditional probabilities
        - Log probabilities
        - Final prediction matches expected class
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
from backend.classifier import (
    NaiveBayesClassifier,
    build_vocabulary,
    text_to_vector,
    transform_documents,
)
from backend.preprocessing import preprocess


# ===========================================================================
# Shared toy dataset (TRD §27)
# ===========================================================================
#
# Spam:
#   doc 0: "free money"   → tokens: [free, money]
#   doc 1: "free prize"   → tokens: [free, prize]
#
# Ham:
#   doc 2: "team meeting" → tokens: [team, meeting]
#   doc 3: "project meeting" → tokens: [project, meeting]
#
# Vocabulary (by order of first appearance):
#   free=0, money=1, prize=2, team=3, meeting=4, project=5
#
# Word counts:
#   spam: free=2, money=1, prize=1, team=0, meeting=0, project=0  → total=4
#   ham:  free=0, money=0, prize=0, team=1, meeting=2, project=1  → total=4
#
# Laplace smoothed P(word|class) with alpha=1, V=6:
#   P(free   | spam) = (2+1)/(4+6) = 3/10 = 0.3
#   P(money  | spam) = (1+1)/(4+6) = 2/10 = 0.2
#   P(prize  | spam) = (1+1)/(4+6) = 2/10 = 0.2
#   P(team   | spam) = (0+1)/(4+6) = 1/10 = 0.1
#   P(meeting| spam) = (0+1)/(4+6) = 1/10 = 0.1
#   P(project| spam) = (0+1)/(4+6) = 1/10 = 0.1
#
#   P(free   | ham)  = (0+1)/(4+6) = 1/10 = 0.1
#   P(money  | ham)  = (0+1)/(4+6) = 1/10 = 0.1
#   P(prize  | ham)  = (0+1)/(4+6) = 1/10 = 0.1
#   P(team   | ham)  = (1+1)/(4+6) = 2/10 = 0.2
#   P(meeting| ham)  = (2+1)/(4+6) = 3/10 = 0.3
#   P(project| ham)  = (1+1)/(4+6) = 2/10 = 0.2
#
# Priors:
#   P(spam) = 2/4 = 0.5  → log = log(0.5) ≈ -0.6931
#   P(ham)  = 2/4 = 0.5  → log = log(0.5) ≈ -0.6931

TOY_DOCS = [
    ["free", "money"],   # spam
    ["free", "prize"],   # spam
    ["team", "meeting"], # ham
    ["project", "meeting"],  # ham
]
TOY_LABELS = ["spam", "spam", "ham", "ham"]

ALPHA = 1.0


def build_toy_model():
    """Build and fit a NaiveBayesClassifier on the toy dataset."""
    vocab = build_vocabulary(TOY_DOCS)
    X = transform_documents(TOY_DOCS, vocab)
    y = np.array(TOY_LABELS)
    model = NaiveBayesClassifier(alpha=ALPHA)
    model.fit(X, y)
    return model, vocab


# ===========================================================================
# __init__
# ===========================================================================

class TestInit:

    def test_default_alpha(self):
        m = NaiveBayesClassifier()
        assert m.alpha == 1.0

    def test_custom_alpha(self):
        m = NaiveBayesClassifier(alpha=0.5)
        assert m.alpha == 0.5

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError):
            NaiveBayesClassifier(alpha=0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError):
            NaiveBayesClassifier(alpha=-1.0)

    def test_not_fitted_at_init(self):
        m = NaiveBayesClassifier()
        assert not m.is_fitted_


# ===========================================================================
# Untrained model
# ===========================================================================

class TestUntrainedModel:

    def test_predict_raises_before_fit(self):
        m = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            m.predict(np.zeros((1, 3), dtype=np.int32))

    def test_predict_one_raises_before_fit(self):
        m = NaiveBayesClassifier()
        with pytest.raises(RuntimeError):
            m.predict_one(np.zeros(3, dtype=np.int32))


# ===========================================================================
# fit() — class counts and priors
# ===========================================================================

class TestFitClassCounts:

    def setup_method(self):
        self.model, self.vocab = build_toy_model()

    def test_classes_detected(self):
        assert set(self.model.classes_) == {"spam", "ham"}

    def test_class_count_spam(self):
        """Spam class has 2 training documents."""
        assert self.model.class_counts_["spam"] == 2

    def test_class_count_ham(self):
        """Ham class has 2 training documents."""
        assert self.model.class_counts_["ham"] == 2

    def test_log_prior_spam(self):
        """log P(spam) = log(2/4) = log(0.5) ≈ -0.6931."""
        expected = math.log(0.5)
        assert abs(self.model.class_log_prior_["spam"] - expected) < 1e-9

    def test_log_prior_ham(self):
        """log P(ham) = log(2/4) = log(0.5) ≈ -0.6931."""
        expected = math.log(0.5)
        assert abs(self.model.class_log_prior_["ham"] - expected) < 1e-9

    def test_is_fitted(self):
        assert self.model.is_fitted_

    def test_vocabulary_size_stored(self):
        assert self.model.vocabulary_size_ == len(self.vocab)


# ===========================================================================
# fit() — word counts
# ===========================================================================

class TestFitWordCounts:

    def setup_method(self):
        self.model, self.vocab = build_toy_model()

    def test_spam_word_count_free(self):
        """'free' appears twice in spam docs."""
        idx = self.vocab["free"]
        assert self.model.word_counts_["spam"][idx] == 2

    def test_spam_word_count_money(self):
        """'money' appears once in spam docs."""
        idx = self.vocab["money"]
        assert self.model.word_counts_["spam"][idx] == 1

    def test_spam_word_count_team_is_zero(self):
        """'team' never appears in spam docs — raw count is 0."""
        idx = self.vocab["team"]
        assert self.model.word_counts_["spam"][idx] == 0

    def test_ham_word_count_meeting(self):
        """'meeting' appears twice in ham docs."""
        idx = self.vocab["meeting"]
        assert self.model.word_counts_["ham"][idx] == 2

    def test_total_words_spam(self):
        """Total word occurrences in spam = 2 (free) + 1 (money) + 1 (prize) = 4."""
        assert self.model.total_words_["spam"] == 4

    def test_total_words_ham(self):
        """Total word occurrences in ham = 1 (team) + 2 (meeting) + 1 (project) = 4."""
        assert self.model.total_words_["ham"] == 4


# ===========================================================================
# fit() — Laplace smoothing and log probabilities
# ===========================================================================

class TestLaplaceSmoothing:

    def setup_method(self):
        self.model, self.vocab = build_toy_model()
        self.V = len(self.vocab)  # 6

    def _expected_log_prob(self, count: int, total: int) -> float:
        """Compute expected log P(word|class) manually."""
        return math.log((count + ALPHA) / (total + ALPHA * self.V))

    def test_smoothed_prob_free_given_spam(self):
        """P(free|spam) = (2+1)/(4+6) = 0.3."""
        idx = self.vocab["free"]
        expected = self._expected_log_prob(2, 4)
        actual = self.model.feature_log_prob_["spam"][idx]
        assert abs(actual - expected) < 1e-9

    def test_smoothed_prob_team_given_spam(self):
        """P(team|spam) = (0+1)/(4+6) = 0.1 — Laplace prevents zero."""
        idx = self.vocab["team"]
        expected = self._expected_log_prob(0, 4)
        actual = self.model.feature_log_prob_["spam"][idx]
        assert abs(actual - expected) < 1e-9

    def test_smoothed_prob_meeting_given_ham(self):
        """P(meeting|ham) = (2+1)/(4+6) = 0.3."""
        idx = self.vocab["meeting"]
        expected = self._expected_log_prob(2, 4)
        actual = self.model.feature_log_prob_["ham"][idx]
        assert abs(actual - expected) < 1e-9

    def test_no_word_has_zero_probability(self):
        """After Laplace smoothing, every log prob must be finite (not -inf)."""
        for c in self.model.classes_:
            assert np.all(np.isfinite(self.model.feature_log_prob_[c])), \
                f"Non-finite log-prob found for class {c!r}"

    def test_zero_count_word_nonzero_probability(self):
        """TRD §38 Test 5: a word with zero training count must have P > 0."""
        # 'free' has 0 count in ham
        idx = self.vocab["free"]
        log_p = self.model.feature_log_prob_["ham"][idx]
        assert log_p > -np.inf
        assert math.exp(log_p) > 0

    def test_feature_log_prob_shape(self):
        """feature_log_prob_ arrays have shape (V,)."""
        V = self.model.vocabulary_size_
        for c in self.model.classes_:
            assert self.model.feature_log_prob_[c].shape == (V,)


# ===========================================================================
# predict() and predict_log_scores()
# ===========================================================================

class TestPredict:

    def setup_method(self):
        self.model, self.vocab = build_toy_model()
        X = transform_documents(TOY_DOCS, self.vocab)
        self.X = X

    def test_predict_shape(self):
        """predict() returns (n_samples,) array."""
        preds = self.model.predict(self.X)
        assert preds.shape == (len(TOY_DOCS),)

    def test_predict_spam_documents(self):
        """First two documents (spam) should be predicted as spam."""
        preds = self.model.predict(self.X)
        assert preds[0] == "spam"
        assert preds[1] == "spam"

    def test_predict_ham_documents(self):
        """Last two documents (ham) should be predicted as ham."""
        preds = self.model.predict(self.X)
        assert preds[2] == "ham"
        assert preds[3] == "ham"

    def test_predict_returns_valid_labels(self):
        """TRD §38 Test 6: each prediction is exactly 'spam' or 'ham'."""
        preds = self.model.predict(self.X)
        for p in preds:
            assert p in ("spam", "ham"), f"Unexpected label: {p!r}"

    def test_predict_one_returns_string(self):
        """predict_one() returns a Python string."""
        x = self.X[0]
        result = self.model.predict_one(x)
        assert isinstance(result, str)
        assert result in ("spam", "ham")

    def test_predict_log_scores_shape(self):
        """predict_log_scores() returns (n_samples, n_classes) array."""
        scores = self.model.predict_log_scores(self.X)
        assert scores.shape == (len(TOY_DOCS), len(self.model.classes_))

    def test_highest_score_matches_prediction(self):
        """The class with the highest log score must equal the prediction."""
        scores = self.model.predict_log_scores(self.X)
        preds = self.model.predict(self.X)
        classes = np.array(self.model.classes_)
        for i in range(len(TOY_DOCS)):
            best_idx = np.argmax(scores[i])
            assert classes[best_idx] == preds[i]


# ===========================================================================
# Mathematical validation (TRD §27) — manually computed expected scores
# ===========================================================================

class TestMathematicalValidation:
    """
    Verify that the model's log scores exactly match hand-computed values.

    Test document: "free money"  (identical to training doc 0)
    Feature vector: [2, 1, 0, 0, 0, 0]  (free=2, money=1, others=0)

    V = 6, alpha = 1.0

    score(spam, doc)
      = log P(spam)
        + count(free)  * log P(free|spam)
        + count(money) * log P(money|spam)
      = log(0.5)
        + 2 * log(3/10)
        + 1 * log(2/10)

    score(ham, doc)
      = log P(ham)
        + count(free)  * log P(free|ham)
        + count(money) * log P(money|ham)
      = log(0.5)
        + 2 * log(1/10)
        + 1 * log(1/10)
    """

    def setup_method(self):
        self.model, self.vocab = build_toy_model()

    def test_spam_score_for_spam_document(self):
        doc = "free money"
        tokens = preprocess(doc)
        x = text_to_vector(tokens, self.vocab)

        # Manual calculation for test doc 'free money':
        # count(free) = 1, count(money) = 1
        expected_spam = (
            math.log(0.5)        # log P(spam)
            + 1 * math.log(3/10) # 1 * log P(free|spam)
            + 1 * math.log(2/10) # 1 * log P(money|spam)
        )

        scores = self.model.get_scores_for_text(tokens, self.vocab)
        assert abs(scores["spam"] - expected_spam) < 1e-9, \
            f"Spam score mismatch: {scores['spam']} vs {expected_spam}"

    def test_ham_score_for_spam_document(self):
        doc = "free money"
        tokens = preprocess(doc)

        # Manual calculation for test doc 'free money':
        # count(free) = 1, count(money) = 1
        expected_ham = (
            math.log(0.5)         # log P(ham)
            + 1 * math.log(1/10)  # 1 * log P(free|ham)
            + 1 * math.log(1/10)  # 1 * log P(money|ham)
        )

        scores = self.model.get_scores_for_text(tokens, self.vocab)
        assert abs(scores["ham"] - expected_ham) < 1e-9

    def test_spam_score_greater_than_ham_for_spam_doc(self):
        """The spam doc must score higher for spam than ham."""
        doc = "free money"
        tokens = preprocess(doc)
        scores = self.model.get_scores_for_text(tokens, self.vocab)
        assert scores["spam"] > scores["ham"]

    def test_ham_score_greater_for_ham_doc(self):
        """A ham doc must score higher for ham than spam."""
        doc = "team meeting"
        tokens = preprocess(doc)
        scores = self.model.get_scores_for_text(tokens, self.vocab)
        assert scores["ham"] > scores["spam"]

    def test_predict_spam_for_spam_input(self):
        """End-to-end: 'free money' must be predicted as spam."""
        tokens = preprocess("free money")
        x = text_to_vector(tokens, self.vocab).reshape(1, -1)
        assert self.model.predict(x)[0] == "spam"

    def test_predict_ham_for_ham_input(self):
        """End-to-end: 'team meeting' must be predicted as ham."""
        tokens = preprocess("team meeting")
        x = text_to_vector(tokens, self.vocab).reshape(1, -1)
        assert self.model.predict(x)[0] == "ham"

    def test_unknown_word_handled_gracefully(self):
        """Unknown words are ignored; prediction still works."""
        tokens = preprocess("free xyzabc")  # 'xyzabc' not in vocab
        x = text_to_vector(tokens, self.vocab).reshape(1, -1)
        pred = self.model.predict(x)[0]
        assert pred in ("spam", "ham")

    def test_empty_input_predict_one(self):
        """All-zero vector (empty input) does not crash; returns a class."""
        x = np.zeros(len(self.vocab), dtype=np.int32)
        pred = self.model.predict_one(x)
        assert pred in ("spam", "ham")


# ===========================================================================
# score()
# ===========================================================================

class TestScore:

    def test_perfect_score_on_toy_training_data(self):
        """
        The model achieves 100% on the 4 tiny training examples.
        (This is expected for a trivially separable small dataset.)
        """
        model, vocab = build_toy_model()
        X = transform_documents(TOY_DOCS, vocab)
        acc = model.score(X, np.array(TOY_LABELS))
        assert acc == 1.0
