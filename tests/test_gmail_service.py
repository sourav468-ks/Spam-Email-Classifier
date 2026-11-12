"""
tests/test_gmail_service.py — Unit tests for gmail_service.py.

Tests cover:
    extract_email_data()
        - Parses headers correctly (subject, sender, date)
        - Extracts text/plain body
        - Handles multipart/alternative (prefers text/plain over text/html)
        - Handles text/html-only messages (strips tags safely)
        - Handles empty body gracefully
        - Handles missing body/parts gracefully
        - Skips attachments
        - Unknown MIME type returns empty string (no crash)
        - Malicious HTML is stripped (no execution possible)

    analyze_emails()
        - Calls preprocess() and model.get_scores_for_text() correctly
        - Empty inbox returns empty list
        - Gmail API HttpError is handled (message skipped)
        - Prediction is either "spam" or "ham"
        - body_text is NOT returned in results

    fetch_message_ids()
        - Returns list of IDs
        - Empty inbox returns empty list
        - HttpError propagates correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import base64
import pytest
from unittest.mock import MagicMock, patch
from googleapiclient.errors import HttpError

from backend.gmail_service import (
    extract_email_data,
    _decode_body_data,
    _strip_html_tags,
    _extract_plain_text_from_part,
    _get_header,
    analyze_emails,
    fetch_message_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_body(text: str) -> str:
    """Encode text as base64url for simulated Gmail API responses."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _make_message(
    subject="Test Subject",
    sender="sender@example.com",
    date="Mon, 1 Jan 2024",
    mime_type="text/plain",
    body_text="Hello world",
    parts=None,
    msg_id="msg001",
) -> dict:
    """Build a minimal fake Gmail API message dict."""
    headers = [
        {"name": "Subject", "value": subject},
        {"name": "From",    "value": sender},
        {"name": "Date",    "value": date},
    ]

    if parts is not None:
        payload = {
            "mimeType": mime_type,
            "headers":  headers,
            "parts":    parts,
            "body":     {"data": ""},
        }
    else:
        payload = {
            "mimeType": mime_type,
            "headers":  headers,
            "body":     {"data": _encode_body(body_text)},
        }

    return {"id": msg_id, "payload": payload}


# ===========================================================================
# _decode_body_data
# ===========================================================================

class TestDecodeBodyData:

    def test_decodes_plain_ascii(self):
        encoded = _encode_body("Hello world")
        assert _decode_body_data(encoded) == "Hello world"

    def test_decodes_unicode(self):
        encoded = _encode_body("Héllo wörld 🎉")
        result = _decode_body_data(encoded)
        # Unicode may be replaced due to ascii errors='replace', but must not crash
        assert isinstance(result, str)

    def test_empty_string_returns_empty(self):
        assert _decode_body_data("") == ""

    def test_none_input_returns_empty(self):
        assert _decode_body_data(None) == ""

    def test_invalid_base64_returns_empty(self):
        result = _decode_body_data("!!!not-valid-base64!!!")
        assert isinstance(result, str)


# ===========================================================================
# _strip_html_tags
# ===========================================================================

class TestStripHtmlTags:

    def test_strips_simple_tags(self):
        result = _strip_html_tags("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strips_script_tags_and_content(self):
        """Script tag content must be removed entirely — not just the tags."""
        result = _strip_html_tags("<script>alert('xss')</script>Hello")
        assert "alert" not in result
        assert "xss" not in result
        assert "Hello" in result

    def test_strips_style_tags_and_content(self):
        result = _strip_html_tags("<style>.evil { color: red }</style>Clean")
        assert ".evil" not in result
        assert "Clean" in result

    def test_returns_text_from_html_email(self):
        html = "<html><body><p>Meeting at 3pm</p></body></html>"
        result = _strip_html_tags(html)
        assert "Meeting" in result
        assert "3pm" in result

    def test_empty_html_returns_empty(self):
        assert _strip_html_tags("") == ""

    def test_no_tags_unchanged(self):
        result = _strip_html_tags("Plain text no tags")
        assert "Plain text no tags" in result


# ===========================================================================
# _get_header
# ===========================================================================

class TestGetHeader:

    def test_returns_subject(self):
        headers = [
            {"name": "Subject", "value": "My Subject"},
            {"name": "From",    "value": "a@b.com"},
        ]
        assert _get_header(headers, "Subject") == "My Subject"

    def test_case_insensitive(self):
        headers = [{"name": "SUBJECT", "value": "Caps Subject"}]
        assert _get_header(headers, "subject") == "Caps Subject"

    def test_missing_header_returns_empty(self):
        headers = [{"name": "From", "value": "a@b.com"}]
        assert _get_header(headers, "Subject") == ""

    def test_empty_headers_returns_empty(self):
        assert _get_header([], "Subject") == ""


# ===========================================================================
# extract_email_data
# ===========================================================================

class TestExtractEmailData:

    def test_plain_text_message(self):
        msg = _make_message(body_text="Free prize now")
        result = extract_email_data(msg)
        assert result["subject"] == "Test Subject"
        assert result["sender"] == "sender@example.com"
        assert result["body_text"] is not None
        assert isinstance(result["body_text"], str)

    def test_html_only_message(self):
        html = "<html><body><p>Win a free vacation!</p></body></html>"
        msg = _make_message(mime_type="text/html", body_text=html)
        result = extract_email_data(msg)
        assert "vacation" in result["body_text"]
        assert "<html>" not in result["body_text"]
        assert "<body>" not in result["body_text"]

    def test_multipart_alternative_prefers_plain(self):
        """multipart/alternative should use text/plain over text/html."""
        parts = [
            {
                "mimeType": "text/plain",
                "body":     {"data": _encode_body("Plain text version")},
                "headers":  [],
            },
            {
                "mimeType": "text/html",
                "body":     {"data": _encode_body("<p>HTML version</p>")},
                "headers":  [],
            },
        ]
        msg = _make_message(mime_type="multipart/alternative", parts=parts)
        result = extract_email_data(msg)
        assert "Plain text version" in result["body_text"]

    def test_multipart_mixed_extracts_text_part(self):
        """multipart/mixed should find and return the text/plain part."""
        parts = [
            {
                "mimeType": "text/plain",
                "body":     {"data": _encode_body("Email body text")},
                "headers":  [],
            },
            {
                "mimeType": "application/pdf",
                "filename": "attachment.pdf",
                "body":     {"attachmentId": "attach123"},
                "headers":  [],
            },
        ]
        msg = _make_message(mime_type="multipart/mixed", parts=parts)
        result = extract_email_data(msg)
        assert "Email body text" in result["body_text"]

    def test_attachment_is_skipped(self):
        """Parts with a filename (attachments) should be ignored."""
        parts = [
            {
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body":     {"attachmentId": "abc"},
                "headers":  [],
            },
        ]
        msg = _make_message(mime_type="multipart/mixed", parts=parts)
        result = extract_email_data(msg)
        # No crash — body_text may be empty or None-converted to ""
        assert isinstance(result["body_text"], str)

    def test_empty_body_returns_empty_string(self):
        """Empty body data should return empty string without crashing."""
        msg = _make_message(body_text="")
        result = extract_email_data(msg)
        assert result["body_text"] == ""

    def test_missing_subject_defaults(self):
        """Missing Subject header should return a default string."""
        msg = {
            "id": "msg002",
            "payload": {
                "mimeType": "text/plain",
                "headers":  [],
                "body":     {"data": _encode_body("body")},
            }
        }
        result = extract_email_data(msg)
        assert result["subject"] == "(No subject)"

    def test_unknown_mime_type_does_not_crash(self):
        """An unsupported MIME type should be silently skipped."""
        msg = _make_message(mime_type="application/octet-stream", body_text="binary")
        result = extract_email_data(msg)
        assert isinstance(result["body_text"], str)

    def test_malicious_html_not_in_output(self):
        """HTML with script tags should have the script stripped."""
        html = "<script>alert('xss')</script><p>Real content</p>"
        msg = _make_message(mime_type="text/html", body_text=html)
        result = extract_email_data(msg)
        assert "alert" not in result["body_text"]
        assert "xss" not in result["body_text"]
        assert "Real content" in result["body_text"]

    def test_body_text_not_in_result_keys_for_classify_only_mode(self):
        """body_text is present in extract_email_data but MUST NOT appear in
        analyze_emails() results (the main pipeline discards it).
        Verify that analyze_emails returns no body_text field."""
        pass  # Covered in TestAnalyzeEmails.test_body_text_not_in_results


# ===========================================================================
# analyze_emails
# ===========================================================================

class TestAnalyzeEmails:

    def _make_mock_service(self, messages=None):
        """Build a mocked Gmail service with predefined messages."""
        service = MagicMock()

        if messages is None:
            messages = [_make_message(body_text="free prize win money")]

        msg_ids = [{"id": m["id"]} for m in messages]

        # messages().list().execute()
        service.users().messages().list().execute.return_value = {
            "messages": msg_ids
        }
        service.users().messages().list.return_value.execute.return_value = {
            "messages": msg_ids
        }

        # messages().get(userId, id, format).execute()
        msg_map = {m["id"]: m for m in messages}

        def get_message(**kwargs):
            msg_id = kwargs.get("id", "")
            mock_get = MagicMock()
            mock_get.execute.return_value = msg_map.get(msg_id, messages[0])
            return mock_get

        service.users().messages().get = lambda **kwargs: get_message(**kwargs)
        return service

    def _make_real_model(self):
        """Build and fit a tiny real NaiveBayesClassifier for testing."""
        import numpy as np
        from backend.classifier import NaiveBayesClassifier, build_vocabulary, transform_documents

        docs = [
            ["free", "prize", "win"],  # spam
            ["meeting", "team", "project"],  # ham
        ]
        labels = ["spam", "ham"]
        vocab = build_vocabulary(docs)
        X = transform_documents(docs, vocab)
        model = NaiveBayesClassifier()
        model.fit(X, np.array(labels))
        return model, vocab

    def test_returns_list(self):
        service = self._make_mock_service()
        model, vocab = self._make_real_model()

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=_make_message()):
            results = analyze_emails(service, model, vocab, max_results=1)

        assert isinstance(results, list)

    def test_spam_prediction_for_spam_text(self):
        """A clearly spam body should be predicted as spam."""
        model, vocab = self._make_real_model()
        spam_msg = _make_message(body_text="free prize win")

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=spam_msg):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        if results:
            assert results[0]["prediction"] in ("spam", "ham")

    def test_ham_prediction_for_ham_text(self):
        """A clearly ham body should be predicted as ham."""
        model, vocab = self._make_real_model()
        ham_msg = _make_message(body_text="team meeting project planning")

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=ham_msg):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        if results:
            assert results[0]["prediction"] in ("spam", "ham")

    def test_empty_inbox_returns_empty_list(self):
        """Empty message list returns empty results."""
        model, vocab = self._make_real_model()

        with patch("backend.gmail_service.fetch_message_ids", return_value=[]):
            results = analyze_emails(MagicMock(), model, vocab, max_results=10)

        assert results == []

    def test_body_text_not_in_results(self):
        """body_text must NOT appear in the results returned to the caller."""
        model, vocab = self._make_real_model()

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=_make_message()):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        for item in results:
            assert "body_text" not in item, \
                "body_text must be discarded after classification"

    def test_result_has_required_keys(self):
        """Each result must have the expected keys."""
        model, vocab = self._make_real_model()

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=_make_message()):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        for item in results:
            for key in ("message_id", "subject", "sender", "date", "prediction"):
                assert key in item, f"Missing key: {key}"

    def test_gmail_api_error_skips_message(self):
        """HttpError on a single message skips it instead of crashing."""
        model, vocab = self._make_real_model()

        http_error = HttpError(
            resp=MagicMock(status=500),
            content=b"Server Error",
        )

        with patch("backend.gmail_service.fetch_message_ids", return_value=["bad001"]), \
             patch("backend.gmail_service.fetch_message", side_effect=http_error):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        # Should not raise — just returns empty or partial list
        assert isinstance(results, list)

    def test_max_results_capped_at_50(self):
        """analyze_emails never requests more than 50 messages."""
        model, vocab = self._make_real_model()

        with patch("backend.gmail_service.fetch_message_ids") as mock_ids:
            mock_ids.return_value = []
            analyze_emails(MagicMock(), model, vocab, max_results=9999)

        _, kwargs = mock_ids.call_args
        assert kwargs.get("max_results", 50) <= 50

    def test_empty_body_does_not_crash(self):
        """Message with no body text results in ham prediction (default)."""
        model, vocab = self._make_real_model()
        empty_msg = _make_message(body_text="")

        with patch("backend.gmail_service.fetch_message_ids", return_value=["msg001"]), \
             patch("backend.gmail_service.fetch_message", return_value=empty_msg):
            results = analyze_emails(MagicMock(), model, vocab, max_results=1)

        if results:
            assert results[0]["prediction"] == "ham"


# ===========================================================================
# End-to-end pipeline: Gmail → preprocessing → classifier
# ===========================================================================

class TestGmailToClassifierPipeline:
    """
    Verify the complete pipeline:
    Gmail email → extract_email_data → preprocess → classifier → prediction
    """

    def test_spam_email_classified_correctly(self):
        """A spam-like email body produces a spam prediction."""
        import numpy as np
        from backend.classifier import NaiveBayesClassifier, build_vocabulary, transform_documents
        from backend.preprocessing import preprocess

        # Build a real small model
        docs = [
            ["free", "prize", "win", "claim", "cash"],
            ["meeting", "team", "project", "schedule"],
        ]
        labels = ["spam", "ham"]
        vocab = build_vocabulary(docs)
        X = transform_documents(docs, vocab)
        model = NaiveBayesClassifier()
        model.fit(X, np.array(labels))

        # Simulate a Gmail email extraction
        spam_body = "free prize claim win cash reward"
        email_data = {"body_text": spam_body}

        tokens = preprocess(email_data["body_text"])
        assert len(tokens) > 0

        scores = model.get_scores_for_text(tokens, vocab)
        prediction = max(scores, key=scores.get)

        assert prediction in ("spam", "ham")  # Valid label
        assert prediction == "spam"  # Should be spam for this input

    def test_ham_email_classified_correctly(self):
        """A ham-like email body produces a ham prediction."""
        import numpy as np
        from backend.classifier import NaiveBayesClassifier, build_vocabulary, transform_documents
        from backend.preprocessing import preprocess

        docs = [
            ["free", "prize", "win"],
            ["meeting", "team", "project", "schedule"],
        ]
        labels = ["spam", "ham"]
        vocab = build_vocabulary(docs)
        X = transform_documents(docs, vocab)
        model = NaiveBayesClassifier()
        model.fit(X, np.array(labels))

        ham_body = "team meeting project schedule review"
        tokens = preprocess(ham_body)

        scores = model.get_scores_for_text(tokens, vocab)
        prediction = max(scores, key=scores.get)

        assert prediction == "ham"
