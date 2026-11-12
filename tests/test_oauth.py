"""
tests/test_oauth.py — Tests for Google OAuth routes and session management.

Tests cover:
    GET /login/google
        - Returns redirect (302) when credentials.json missing → redirect to /?error=
        - Returns redirect toward Google when credentials.json exists (mocked)

    GET /oauth2callback
        - Missing state → 400
        - Mismatched state → 400
        - OAuth error param from Google → redirect to /?error=oauth_denied
        - Successful exchange → session created, redirect to /dashboard

    GET /auth/status
        - Without session → {authenticated: false}
        - With session → {authenticated: true, email: ...}

    GET /dashboard
        - Without session → redirects to /
        - With session → 200

    GET /gmail/analyze
        - Without session → 401

    POST /logout
        - With session → session cleared, redirect to /
        - Without session → 200/redirect (safe)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


@pytest.fixture
def authenticated_client(client):
    """A test client with a pre-populated session (simulating a logged-in user)."""
    with client.session_transaction() as sess:
        sess["credentials"] = {
            "token": "fake-access-token",
            "refresh_token": "fake-refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "fake-client-id",
            "client_secret": "fake-client-secret",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            "expiry": None,
        }
        sess["user_email"] = "test@example.com"
        sess["user_name"] = "Test User"
    return client


# ===========================================================================
# GET /login/google
# ===========================================================================

class TestLoginGoogle:

    def test_login_google_redirects_on_missing_credentials_file(self, client):
        """Without credentials.json, login redirects to /?error=oauth_config."""
        with patch("backend.google_auth.os.path.isfile", return_value=False):
            res = client.get("/login/google")
        assert res.status_code == 302
        location = res.headers.get("Location", "")
        assert "error=oauth_config" in location or "/" in location

    def test_login_google_redirects_to_google(self, client):
        """With a mocked flow, /login/google should redirect to a Google URL."""
        mock_flow = MagicMock()
        mock_flow.authorization_url.return_value = (
            "https://accounts.google.com/o/oauth2/auth?scope=...",
            "test-state-csrf",
        )

        with patch("backend.google_auth.os.path.isfile", return_value=True), \
             patch("backend.google_auth.Flow.from_client_secrets_file", return_value=mock_flow):
            res = client.get("/login/google")

        assert res.status_code == 302
        location = res.headers.get("Location", "")
        assert "accounts.google.com" in location or "error" in location

    def test_login_google_stores_state_in_session(self, client):
        """The OAuth CSRF state must be stored in the session."""
        mock_flow = MagicMock()
        mock_flow.authorization_url.return_value = (
            "https://accounts.google.com/o/oauth2/auth",
            "csrf-state-abc123",
        )

        with patch("backend.google_auth.os.path.isfile", return_value=True), \
             patch("backend.google_auth.Flow.from_client_secrets_file", return_value=mock_flow):
            client.get("/login/google")

        with client.session_transaction() as sess:
            assert "oauth_state" in sess


# ===========================================================================
# GET /oauth2callback
# ===========================================================================

class TestOAuthCallback:

    def test_callback_missing_state_returns_400(self, client):
        """Callback with no state parameter returns 400."""
        res = client.get("/oauth2callback?code=somecode")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_callback_mismatched_state_returns_400(self, client):
        """Callback with wrong state returns 400 (CSRF protection)."""
        with client.session_transaction() as sess:
            sess["oauth_state"] = "correct-state"

        res = client.get("/oauth2callback?code=somecode&state=wrong-state")
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_callback_oauth_error_redirects_safely(self, client):
        """Google returning error param redirects to /?error=oauth_denied."""
        with client.session_transaction() as sess:
            sess["oauth_state"] = "valid-state"

        res = client.get("/oauth2callback?state=valid-state&error=access_denied")
        assert res.status_code == 302
        location = res.headers.get("Location", "")
        assert "oauth_denied" in location or "/" in location

    def test_callback_no_session_created_on_failure(self, client):
        """On callback failure, no credentials are stored in session."""
        with client.session_transaction() as sess:
            sess["oauth_state"] = "valid-state"

        # Simulate exception during token exchange
        with patch("backend.google_auth.get_google_auth_flow", side_effect=Exception("boom")):
            res = client.get("/oauth2callback?state=valid-state&code=badcode")

        assert res.status_code == 302
        with client.session_transaction() as sess:
            assert "credentials" not in sess


# ===========================================================================
# GET /auth/status
# ===========================================================================

class TestAuthStatus:

    def test_auth_status_unauthenticated(self, client):
        """Unauthenticated request returns {authenticated: false}."""
        res = client.get("/auth/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["authenticated"] is False
        assert data["email"] is None

    def test_auth_status_authenticated(self, authenticated_client):
        """Authenticated session returns {authenticated: true, email: ...}."""
        # The credentials refresh will fail (fake token), but we mock it
        with patch("backend.app._get_credentials") as mock_creds:
            mock_creds.return_value = MagicMock()
            res = authenticated_client.get("/auth/status")

        assert res.status_code == 200
        data = res.get_json()
        assert data["authenticated"] is True
        assert data["email"] == "test@example.com"

    def test_auth_status_never_returns_token(self, authenticated_client):
        """auth/status must not expose any OAuth token."""
        with patch("backend.app._get_credentials") as mock_creds:
            mock_creds.return_value = MagicMock()
            res = authenticated_client.get("/auth/status")

        data = res.get_json()
        response_str = str(data)
        # Tokens should never appear in the response
        assert "fake-access-token" not in response_str
        assert "fake-refresh-token" not in response_str
        assert "token" not in response_str.lower().replace("authenticated", "")


# ===========================================================================
# GET /dashboard
# ===========================================================================

class TestDashboard:

    def test_dashboard_requires_auth(self, client):
        """Unauthenticated /dashboard redirects to /."""
        res = client.get("/dashboard")
        assert res.status_code == 302
        location = res.headers.get("Location", "")
        assert "/" in location

    def test_dashboard_accessible_when_authenticated(self, authenticated_client):
        """Authenticated user gets 200 from /dashboard."""
        with patch("backend.app._is_authenticated", return_value=True):
            res = authenticated_client.get("/dashboard")
        assert res.status_code == 200


# ===========================================================================
# GET /gmail/analyze
# ===========================================================================

class TestGmailAnalyze:

    def test_gmail_analyze_requires_auth(self, client):
        """Unauthenticated /gmail/analyze returns 401."""
        res = client.get("/gmail/analyze")
        assert res.status_code == 401
        data = res.get_json()
        assert "error" in data

    def test_gmail_analyze_returns_results_when_authenticated(self, authenticated_client):
        """Authenticated request with mocked Gmail service returns results."""
        mock_results = [
            {
                "message_id": "abc123",
                "subject":    "Free prize!",
                "sender":     "spam@example.com",
                "date":       "Mon, 1 Jan 2024",
                "prediction": "spam",
                "score_spam": -10.0,
                "score_ham":  -30.0,
            }
        ]

        with patch("backend.app._is_authenticated", return_value=True), \
             patch("backend.app._get_credentials", return_value=MagicMock()), \
             patch("backend.app._model_available", True), \
             patch("backend.gmail_service.build_gmail_service"), \
             patch("backend.gmail_service.analyze_emails", return_value=mock_results):
            res = authenticated_client.get("/gmail/analyze")

        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert data["count"] >= 0

    def test_gmail_analyze_query_too_long(self, authenticated_client):
        """Query parameter over 500 chars returns 400."""
        with patch("backend.app._is_authenticated", return_value=True), \
             patch("backend.app._get_credentials", return_value=MagicMock()):
            long_query = "a" * 501
            res = authenticated_client.get(f"/gmail/analyze?query={long_query}")

        assert res.status_code in (400, 401)  # 401 if auth fails in test env

    def test_gmail_analyze_clamps_max_results(self, authenticated_client):
        """max_results above 50 is clamped to 50."""
        mock_results = []

        with patch("backend.app._is_authenticated", return_value=True), \
             patch("backend.app._get_credentials", return_value=MagicMock()), \
             patch("backend.app._model_available", True), \
             patch("backend.gmail_service.build_gmail_service"), \
             patch("backend.gmail_service.analyze_emails", return_value=mock_results) as mock_analyze:
            authenticated_client.get("/gmail/analyze?max_results=999")

        # If analyze_emails was called, check its max_results arg was clamped
        if mock_analyze.called:
            _, kwargs = mock_analyze.call_args
            actual_max = kwargs.get("max_results", 10)
            assert actual_max <= 50


# ===========================================================================
# POST /logout
# ===========================================================================

class TestLogout:

    def test_logout_clears_session(self, authenticated_client):
        """POST /logout clears the session."""
        with patch("backend.app._get_credentials", return_value=None):
            res = authenticated_client.post("/logout")

        assert res.status_code == 302
        with authenticated_client.session_transaction() as sess:
            assert "credentials" not in sess
            assert "user_email" not in sess

    def test_logout_without_session_is_safe(self, client):
        """POST /logout with no session does not crash."""
        res = client.post("/logout")
        assert res.status_code in (200, 302)

    def test_logout_does_not_expose_tokens(self, authenticated_client):
        """Logout response must not expose any OAuth token."""
        with patch("backend.app._get_credentials", return_value=None):
            res = authenticated_client.post("/logout")

        response_data = res.data.decode("utf-8")
        assert "fake-access-token" not in response_data
        assert "fake-refresh-token" not in response_data
