"""
tests/test_preprocessing.py — Unit tests for backend/preprocessing.py

Tests cover:
    clean_text()
        - Lowercase conversion
        - Punctuation removal
        - Digit removal
        - Unicode / accented characters
        - Whitespace normalization
        - None / non-string input
        - Empty string

    tokenize()
        - Normal input
        - Empty string
        - Multi-space input (already cleaned)

    preprocess()
        - Full pipeline
        - Edge cases: empty, None, all-punctuation, unicode

    preprocess_batch()
        - Batch processing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.preprocessing import clean_text, tokenize, preprocess, preprocess_batch


# =============================================================================
# clean_text()
# =============================================================================

class TestCleanText:

    def test_lowercase(self):
        """Uppercase letters become lowercase."""
        assert clean_text("FREE OFFER") == "free offer"

    def test_mixed_case(self):
        """Mixed case is fully lowercased."""
        assert clean_text("Free Offer WIN") == "free offer win"

    def test_punctuation_removed(self):
        """Punctuation characters are replaced with spaces and collapsed."""
        assert clean_text("FREE!!!") == "free"

    def test_dollar_signs_removed(self):
        """Currency symbols are treated as non-alpha."""
        assert clean_text("WIN $$$") == "win"

    def test_digits_removed(self):
        """Digits are treated as non-alphabetic."""
        result = clean_text("Call 1800 now")
        assert "1800" not in result
        assert "call" in result
        assert "now" in result

    def test_unicode_normalized(self):
        """Accented characters are stripped via ASCII encoding."""
        result = clean_text("Héllo Wörld")
        assert result == "hello world"

    def test_whitespace_normalized(self):
        """Multiple spaces become one space; leading/trailing stripped."""
        assert clean_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert clean_text("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input returns empty string."""
        assert clean_text("   \t\n  ") == ""

    def test_none_input(self):
        """None input is handled without raising an exception."""
        result = clean_text(None)
        assert isinstance(result, str)

    def test_all_punctuation(self):
        """A string of only punctuation yields empty string."""
        assert clean_text("!!!???$$$!!!") == ""

    def test_preserves_spaces_between_words(self):
        """Words separated by any non-alpha chars still produce separate tokens."""
        result = clean_text("free,offer,win")
        # Commas → spaces → collapsed → 'free offer win'
        assert result == "free offer win"

    def test_numbers_as_words(self):
        """'FREE1' → 'free' (digit stripped)."""
        assert clean_text("FREE1") == "free"

    def test_html_stripped(self):
        """HTML tags have their angle brackets removed; text content survives."""
        result = clean_text("<b>free</b>")
        assert "free" in result
        # No angle brackets or tag names remain as-is
        assert "<" not in result


# =============================================================================
# tokenize()
# =============================================================================

class TestTokenize:

    def test_splits_on_spaces(self):
        """Splits clean text into tokens on whitespace."""
        assert tokenize("free win now") == ["free", "win", "now"]

    def test_single_token(self):
        assert tokenize("free") == ["free"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_no_empty_tokens(self):
        """After clean_text, there should be no empty-string tokens."""
        # tokenize alone on single-space input should produce no empty strings
        result = tokenize("hello world")
        assert "" not in result


# =============================================================================
# preprocess()
# =============================================================================

class TestPreprocess:

    def test_standard_spam_example(self):
        """PRD acceptance test: 'FREE!!! WIN $$$' → ['free', 'win']."""
        assert preprocess("FREE!!! WIN $$$") == ["free", "win"]

    def test_standard_ham_example(self):
        """A normal sentence tokenizes correctly."""
        result = preprocess("Can we meet tomorrow at 10 AM?")
        assert "meet" in result
        assert "tomorrow" in result

    def test_empty_input_returns_empty_list(self):
        assert preprocess("") == []

    def test_none_returns_empty_list(self):
        assert preprocess(None) == []

    def test_whitespace_only_returns_empty_list(self):
        assert preprocess("    ") == []

    def test_all_punctuation_returns_empty_list(self):
        assert preprocess("!!! $$$ ???") == []

    def test_lowercase_output(self):
        """All output tokens must be lowercase."""
        result = preprocess("HELLO WORLD")
        assert all(t == t.lower() for t in result)

    def test_no_punctuation_in_output(self):
        """No token should contain punctuation characters."""
        result = preprocess("free!!! offer??? win$$$ now.")
        for token in result:
            assert token.isalpha(), f"Non-alpha token: {token!r}"

    def test_unicode_input(self):
        """Accented characters are stripped without crashing."""
        result = preprocess("Héllo Wörld café")
        assert isinstance(result, list)
        # 'hello', 'world', 'caf' are plausible results
        assert len(result) > 0

    def test_very_long_input_no_crash(self):
        """Very long input is processed without error."""
        long_text = "free offer win prize money " * 1000
        result = preprocess(long_text)
        assert len(result) > 0

    def test_deterministic(self):
        """Calling preprocess twice with the same input gives the same result."""
        text = "Congratulations! You won a free prize!!!"
        assert preprocess(text) == preprocess(text)


# =============================================================================
# preprocess_batch()
# =============================================================================

class TestPreprocessBatch:

    def test_returns_list_of_lists(self):
        texts = ["free prize", "team meeting"]
        result = preprocess_batch(texts)
        assert isinstance(result, list)
        assert all(isinstance(r, list) for r in result)

    def test_length_preserved(self):
        texts = ["free", "prize", "meet"]
        result = preprocess_batch(texts)
        assert len(result) == len(texts)

    def test_empty_batch(self):
        assert preprocess_batch([]) == []

    def test_each_element_matches_preprocess(self):
        texts = ["FREE!!! WIN $$$", "Can we meet?", ""]
        result = preprocess_batch(texts)
        for i, text in enumerate(texts):
            assert result[i] == preprocess(text)
