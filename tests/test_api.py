"""
tests/test_api.py — Unit and integration tests for the Flask API (backend/app.py).

Tests cover:
    GET /               Returns HTTP 200 and frontend HTML
    GET /health         Returns HTTP 200 with model status
    POST /predict
        - Valid spam message -> returns prediction 'spam' with scores
        - Valid ham message  -> returns prediction 'ham' with scores
        - Empty message string -> returns HTTP 400 Bad Request
        - Whitespace-only string -> returns HTTP 400 Bad Request
        - Non-string message (e.g. integer) -> returns HTTP 400 Bad Request
        - Missing 'message' field -> returns HTTP 400 Bad Request
        - Non-JSON Content-Type -> returns HTTP 400 Bad Request
        - Oversized message (> 10,000 chars) -> returns HTTP 400 Bad Request
        - HTML/Script injection in message -> treated as plain text, no XSS/crash
        - SQL injection string in message -> treated as plain text, no SQL error
        - Path traversal string in message -> treated as plain text, no file access
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json
from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAppEndpoints:

    def test_get_index(self, client):
        """GET / returns HTML page with status 200."""
        res = client.get("/")
        assert res.status_code == 200
        assert b"SpamGuard" in res.data or b"html" in res.data

    def test_get_health(self, client):
        """GET /health returns JSON with status 'ok' and model_loaded."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["vocabulary_size"] > 0


class TestPredictEndpoint:

    def test_predict_spam(self, client):
        """Standard spam message receives 'spam' prediction."""
        payload = {"message": "WINNER! You have won a $1,000,000 cash prize! Claim NOW!"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["prediction"] == "spam"
        assert "score_spam" in data
        assert "score_ham" in data
        assert data["score_spam"] > data["score_ham"]

    def test_predict_ham(self, client):
        """Standard legitimate message receives 'ham' prediction."""
        payload = {"message": "Hi Alice, could we reschedule our meeting to 3 PM tomorrow?"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["prediction"] == "ham"
        assert "score_spam" in data
        assert "score_ham" in data
        assert data["score_ham"] > data["score_spam"]

    def test_predict_empty_message(self, client):
        """Empty message rejected with HTTP 400."""
        res = client.post("/predict", json={"message": ""})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_predict_whitespace_only(self, client):
        """Whitespace-only message rejected with HTTP 400."""
        res = client.post("/predict", json={"message": "    \n\t  "})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_predict_missing_message_field(self, client):
        """Missing 'message' key rejected with HTTP 400."""
        res = client.post("/predict", json={"text": "Hello"})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_predict_non_string_message(self, client):
        """Non-string 'message' (e.g. integer) rejected with HTTP 400."""
        res = client.post("/predict", json={"message": 12345})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_predict_non_json_request(self, client):
        """Non-JSON payload rejected with HTTP 400."""
        res = client.post("/predict", data="message=hello", content_type="application/x-www-form-urlencoded")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_predict_oversized_message(self, client):
        """Message exceeding 10,000 characters is rejected with HTTP 400."""
        large_msg = "free prize money " * 1000  # ~17,000 chars
        res = client.post("/predict", json={"message": large_msg})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data
        assert "maximum length" in data["error"].lower()


class TestSecurityPayloads:

    def test_xss_payload(self, client):
        """HTML / script tag is sanitized and safely classified without execution or crash."""
        payload = {"message": "<script>alert('XSS')</script> WIN FREE MONEY NOW"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["prediction"] in ("spam", "ham")

    def test_sql_injection_payload(self, client):
        """SQL injection attempt is treated as plain text."""
        payload = {"message": "'; DROP TABLE users; -- WIN PRIZE"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["prediction"] in ("spam", "ham")

    def test_path_traversal_payload(self, client):
        """Path traversal attempt is treated as plain text."""
        payload = {"message": "../../../etc/passwd .../model.pkl WIN PRIZE"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["prediction"] in ("spam", "ham")


class TestNewAuthEndpoints:
    """Tests for the new OAuth endpoints added to app.py.

    Verifies that existing /predict functionality is not broken by the OAuth
    changes, and that new endpoints behave correctly when unauthenticated.
    """

    def test_auth_status_without_session(self, client):
        """GET /auth/status without a session returns authenticated=false."""
        res = client.get("/auth/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["authenticated"] is False
        assert data["email"] is None

    def test_dashboard_without_session_redirects(self, client):
        """GET /dashboard without a session redirects (302) to /."""
        res = client.get("/dashboard")
        assert res.status_code == 302
        location = res.headers.get("Location", "")
        assert "/" in location

    def test_gmail_analyze_without_session_returns_401(self, client):
        """GET /gmail/analyze without a session returns 401."""
        res = client.get("/gmail/analyze")
        assert res.status_code == 401
        data = res.get_json()
        assert "error" in data

    def test_predict_still_works_without_auth(self, client):
        """POST /predict must still work without a session (no auth required)."""
        res = client.post("/predict", json={"message": "free prize money win claim"})
        assert res.status_code == 200
        data = res.get_json()
        assert "prediction" in data

    def test_logout_without_session_is_safe(self, client):
        """POST /logout without a session does not crash."""
        res = client.post("/logout")
        assert res.status_code in (200, 302)

    def test_auth_status_never_contains_tokens(self, client):
        """auth/status must not expose access_token or refresh_token in response."""
        res = client.get("/auth/status")
        response_text = res.data.decode("utf-8")
        assert "access_token" not in response_text
        assert "refresh_token" not in response_text
        assert "client_secret" not in response_text
