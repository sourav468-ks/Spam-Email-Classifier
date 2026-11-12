"""
google_auth.py — Google OAuth 2.0 flow helpers for SpamGuard.

This module is the sole place that knows about the OAuth flow.
It never touches Gmail API operations (those live in gmail_service.py)
and never touches Flask routing (that lives in app.py).

Security notes
--------------
* credentials.json is read from a fixed path defined in config.py.
  The path is never derived from user input.
* Credentials are serialised to a plain dict for storage in the Flask
  session (server-side cookie).  The dict contains an access_token and
  optionally a refresh_token.  Neither is ever logged or returned to
  the browser.
* The OAuth state parameter (CSRF protection) is stored in the Flask
  session and validated in the callback before the code is exchanged.
* Token revocation is best-effort: if it fails we still clear the
  local session so the user is effectively logged out.
"""

import os
import logging

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import google.auth.exceptions

from backend.config import (
    GOOGLE_CREDENTIALS_FILE,
    GOOGLE_SCOPES,
    OAUTH_REDIRECT_URI,
    FLASK_DEBUG,
)

logger = logging.getLogger(__name__)

# Allow OAuth 2.0 over HTTP for local development ONLY.
# Without this, requests-oauthlib throws InsecureTransportError when exchanging the token.
if "localhost" in OAUTH_REDIRECT_URI or "127.0.0.1" in OAUTH_REDIRECT_URI or FLASK_DEBUG:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


# ---------------------------------------------------------------------------
# Flow factory
# ---------------------------------------------------------------------------

def get_google_auth_flow(state: str = None) -> Flow:
    """Return a configured OAuth 2.0 Flow object.

    The Flow is created from the credentials.json downloaded from Google
    Cloud Console.  The redirect_uri must exactly match the URI configured
    in the Google Cloud OAuth client.

    Raises
    ------
    FileNotFoundError
        If credentials.json is missing.
    ValueError
        If OAUTH_REDIRECT_URI is not set.
    """
    if not os.path.isfile(GOOGLE_CREDENTIALS_FILE):
        raise FileNotFoundError(
            f"credentials.json not found at: {GOOGLE_CREDENTIALS_FILE}\n"
            "Download it from Google Cloud Console → APIs & Services → "
            "Credentials and place it at the project root."
        )

    if not OAUTH_REDIRECT_URI:
        raise ValueError(
            "OAUTH_REDIRECT_URI is not set. "
            "Add it to your .env file, e.g.: "
            "OAUTH_REDIRECT_URI=http://localhost:5000/oauth2callback"
        )

    flow = Flow.from_client_secrets_file(
        GOOGLE_CREDENTIALS_FILE,
        scopes=GOOGLE_SCOPES,
        redirect_uri=OAUTH_REDIRECT_URI,
        state=state,
    )
    return flow


def get_authorization_url(flow: Flow) -> tuple[str, str, str]:
    """Generate the Google authorization URL, CSRF state token, and PKCE verifier.

    Parameters
    ----------
    flow : Flow
        An OAuth Flow object from :func:`get_google_auth_flow`.

    Returns
    -------
    (authorization_url, state, code_verifier) : tuple[str, str, str]
        * authorization_url — redirect the user here.
        * state — store in session for CSRF validation.
        * code_verifier — store in session for PKCE validation.
    """
    authorization_url, state = flow.authorization_url(
        # Force consent screen every time so the user sees what they grant.
        # This also ensures refresh_token is included in the response.
        prompt="consent",
        # Request offline access to get a refresh token.
        access_type="offline",
        # Include granted scopes in the response for transparency.
        include_granted_scopes="true",
    )
    code_verifier = getattr(flow, "code_verifier", None)
    return authorization_url, state, code_verifier


def exchange_code_for_credentials(
    flow: Flow,
    authorization_response: str,
) -> Credentials:
    """Exchange the authorization code for OAuth credentials.

    Parameters
    ----------
    flow : Flow
        The same Flow object used to generate the auth URL.
        It must have the correct state set.
    authorization_response : str
        The full callback URL including the code and state query parameters.
        Example: "http://localhost:5000/oauth2callback?code=...&state=..."

    Returns
    -------
    google.oauth2.credentials.Credentials
        Contains access_token and (if access_type=offline) refresh_token.
    """
    flow.fetch_token(authorization_response=authorization_response)
    return flow.credentials


# ---------------------------------------------------------------------------
# Session serialisation helpers
# ---------------------------------------------------------------------------

def credentials_to_dict(credentials: Credentials) -> dict:
    """Serialise credentials to a plain dict safe for session storage.

    The dict is stored in the Flask session (an encrypted cookie).
    It contains the access_token and refresh_token (both sensitive).
    They are NEVER logged or returned to the browser.

    Parameters
    ----------
    credentials : google.oauth2.credentials.Credentials

    Returns
    -------
    dict
        Keys: token, refresh_token, token_uri, client_id, client_secret,
              scopes, expiry (ISO string or None).
    """
    return {
        "token":         credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri":     credentials.token_uri,
        "client_id":     credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes":        list(credentials.scopes) if credentials.scopes else [],
        "expiry":        (
            credentials.expiry.isoformat()
            if credentials.expiry else None
        ),
    }


def credentials_from_dict(data: dict) -> Credentials:
    """Deserialise credentials from a session dict.

    Parameters
    ----------
    data : dict
        As returned by :func:`credentials_to_dict`.

    Returns
    -------
    google.oauth2.credentials.Credentials
    """
    import datetime

    expiry = None
    if data.get("expiry"):
        try:
            expiry = datetime.datetime.fromisoformat(data["expiry"])
        except (ValueError, TypeError):
            expiry = None

    return Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
        expiry=expiry,
    )


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def refresh_credentials_if_needed(credentials: Credentials) -> Credentials:
    """Refresh expired credentials using the stored refresh token.

    Parameters
    ----------
    credentials : Credentials

    Returns
    -------
    Credentials (possibly updated with a new access token)

    Raises
    ------
    google.auth.exceptions.RefreshError
        If the refresh token is invalid or has been revoked.
    """
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            logger.info("OAuth credentials refreshed successfully.")
        except google.auth.exceptions.RefreshError:
            logger.warning("Failed to refresh OAuth credentials.")
            raise
    return credentials


# ---------------------------------------------------------------------------
# Token revocation
# ---------------------------------------------------------------------------

def revoke_credentials(credentials: Credentials) -> bool:
    """Revoke the OAuth token at Google's revocation endpoint.

    Best-effort: if revocation fails (e.g. already revoked), we log a
    warning and return False.  The caller should still clear the local
    session regardless.

    Parameters
    ----------
    credentials : Credentials

    Returns
    -------
    bool
        True if revocation succeeded, False otherwise.
    """
    import requests as _requests

    token = credentials.token
    if not token:
        logger.warning("No token to revoke.")
        return False

    try:
        response = _requests.post(
            "https://oauth2.googleapis.com/revoke",
            params={"token": token},
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=5,
        )
        if response.status_code == 200:
            logger.info("OAuth token revoked successfully.")
            return True
        else:
            logger.warning(
                "Token revocation returned status %d.", response.status_code
            )
            return False
    except Exception:
        logger.warning("Token revocation request failed.", exc_info=True)
        return False
