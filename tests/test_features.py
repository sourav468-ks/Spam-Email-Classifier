"""
tests/test_features.py — Unit tests for vocabulary and Bag-of-Words in classifier.py

Tests cover:
    build_vocabulary()
        - Correct index assignment
        - Duplicate tokens get one index
        - Empty input
        - Order of first occurrence

    text_to_vector()
        - Correct word counts
        - Unknown words ignored
        - Empty token list → zero vector
        - Vector length equals vocabulary size

    transform_documents()
        - Shape correctness
        - Row-per-document semantics
        - Consistent with text_to_vector

    TRD Acceptance Tests (§38):
        - Test 2: vocabulary from "free prize", "free money" → 3 unique tokens
        - Test 3: "free free prize" with vocab → [2, 1]
        - Test 4: "free xyzabc" with vocab → [1, 0]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from backend.classifier import build_vocabulary, text_to_vector, transform_documents


# =============================================================================
# build_vocabulary()
# =============================================================================

class TestBuildVocabulary:

    def test_basic_indexing(self):
        """Each unique token gets a unique integer starting from 0."""
        vocab = build_vocabulary([["free", "prize"]])
        assert vocab["free"] == 0
        assert vocab["prize"] == 1

    def test_duplicate_words_one_index(self):
        """Repeated tokens across documents get exactly one index."""
        vocab = build_vocabulary([["free", "prize"], ["free", "money"]])
        assert "free" in vocab
        # 'free' appears in both docs but only one entry
        assert len([k for k in vocab if k == "free"]) == 1

    def test_three_unique_tokens(self):
        """TRD §38 Test 2: 'free prize' + 'free money' → 3 unique tokens."""
        vocab = build_vocabulary([["free", "prize"], ["free", "money"]])
        assert set(vocab.keys()) == {"free", "prize", "money"}
        assert len(vocab) == 3

    def test_empty_input(self):
        """Empty list of token lists produces empty vocabulary."""
        vocab = build_vocabulary([])
        assert vocab == {}

    def test_empty_documents(self):
        """Token lists that are themselves empty produce nothing."""
        vocab = build_vocabulary([[], [], []])
        assert vocab == {}

    def test_mixed_empty_and_nonempty(self):
        """Empty docs don't break the index for subsequent docs."""
        vocab = build_vocabulary([[], ["free"], [], ["prize"]])
        assert len(vocab) == 2
        assert "free" in vocab
        assert "prize" in vocab

    def test_index_contiguous_from_zero(self):
        """Indices are 0, 1, 2, ... without gaps."""
        vocab = build_vocabulary([["a", "b", "c"]])
        assert sorted(vocab.values()) == [0, 1, 2]

    def test_first_occurrence_order(self):
        """The first token seen gets index 0, next new token gets 1, etc."""
        vocab = build_vocabulary([["alpha", "beta", "gamma"]])
        assert vocab["alpha"] == 0
        assert vocab["beta"]  == 1
        assert vocab["gamma"] == 2

    def test_intra_document_duplicates(self):
        """Duplicate tokens within the same document don't inflate vocab."""
        vocab = build_vocabulary([["free", "free", "free"]])
        assert len(vocab) == 1
        assert vocab["free"] == 0

    def test_large_vocabulary(self):
        """Large input produces correct number of unique entries."""
        words = [f"word{i}" for i in range(100)]
        vocab = build_vocabulary([words])
        assert len(vocab) == 100


# =============================================================================
# text_to_vector()
# =============================================================================

class TestTextToVector:

    def setup_method(self):
        """Small vocabulary fixture used across tests."""
        self.vocab = {"free": 0, "prize": 1, "meeting": 2}

    def test_basic_counts(self):
        """TRD §38 Test 3: 'free free prize' → [2, 1, 0]."""
        tokens = ["free", "free", "prize"]
        vec = text_to_vector(tokens, self.vocab)
        np.testing.assert_array_equal(vec, [2, 1, 0])

    def test_unknown_word_ignored(self):
        """TRD §38 Test 4: 'free xyzabc' → [1, 0, 0]."""
        tokens = ["free", "xyzabc"]
        vec = text_to_vector(tokens, self.vocab)
        np.testing.assert_array_equal(vec, [1, 0, 0])

    def test_empty_tokens_zero_vector(self):
        """Empty token list produces an all-zeros vector."""
        vec = text_to_vector([], self.vocab)
        np.testing.assert_array_equal(vec, [0, 0, 0])

    def test_vector_length_equals_vocab_size(self):
        """Output length always equals vocabulary size."""
        vec = text_to_vector(["free"], self.vocab)
        assert len(vec) == len(self.vocab)

    def test_dtype_is_int(self):
        """Vector dtype must be integer (word counts)."""
        vec = text_to_vector(["free"], self.vocab)
        assert np.issubdtype(vec.dtype, np.integer)

    def test_all_zeros_for_unknown_input(self):
        """All unknown words → zero vector."""
        vec = text_to_vector(["abc", "xyz", "qrs"], self.vocab)
        np.testing.assert_array_equal(vec, [0, 0, 0])

    def test_only_meeting_word(self):
        """'meeting' increments index 2, others stay 0."""
        vec = text_to_vector(["meeting"], self.vocab)
        np.testing.assert_array_equal(vec, [0, 0, 1])

    def test_repeated_word_high_count(self):
        """Repetition correctly accumulates counts."""
        vec = text_to_vector(["free"] * 10, self.vocab)
        assert vec[0] == 10

    def test_empty_vocabulary(self):
        """With an empty vocabulary, output is a zero-length array."""
        vec = text_to_vector(["free"], {})
        assert vec.shape == (0,)


# =============================================================================
# transform_documents()
# =============================================================================

class TestTransformDocuments:

    def setup_method(self):
        self.vocab = {"free": 0, "prize": 1, "meeting": 2}

    def test_shape(self):
        """Output shape is (n_docs, vocab_size)."""
        docs = [["free", "prize"], ["meeting"]]
        X = transform_documents(docs, self.vocab)
        assert X.shape == (2, 3)

    def test_single_document(self):
        """Single document produces a (1, V) matrix."""
        X = transform_documents([["free", "free"]], self.vocab)
        assert X.shape == (1, 3)
        assert X[0, 0] == 2

    def test_row_matches_text_to_vector(self):
        """Each row must match the output of text_to_vector for that document."""
        docs = [["free", "prize"], ["free", "free", "meeting"]]
        X = transform_documents(docs, self.vocab)
        for i, tokens in enumerate(docs):
            expected = text_to_vector(tokens, self.vocab)
            np.testing.assert_array_equal(X[i], expected)

    def test_empty_document_list(self):
        """Empty input produces a (0, V) matrix."""
        X = transform_documents([], self.vocab)
        assert X.shape == (0, len(self.vocab))

    def test_known_values(self):
        """Known example: verify exact matrix values."""
        docs = [["free", "free"], ["prize"]]
        X = transform_documents(docs, self.vocab)
        np.testing.assert_array_equal(X, [[2, 0, 0], [0, 1, 0]])
