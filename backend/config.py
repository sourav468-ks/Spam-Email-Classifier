"""
config.py — Central configuration constants for the Spam Email Classifier.

All tunable parameters live here so no magic numbers are scattered
through the codebase.  Import from here rather than hard-coding values.

Environment variables
---------------------
Sensitive values (OAuth credentials, Flask secret key) are loaded from
the environment.  For local development, place them in a .env file and
install python-dotenv.  The .env file must NEVER be committed to Git.

Required env vars for Gmail OAuth:
    FLASK_SECRET_KEY        — random secret for Flask session signing
    GOOGLE_REDIRECT_URI     — must match Google Cloud OAuth client config
                              (default: http://localhost:5000/oauth2callback)
    GMAIL_MAX_RESULTS       — optional, defaults to 10

The Google OAuth client credentials are read from credentials.json at the
project root (downloaded from Google Cloud Console).
"""

import os

# Load .env file if present (development convenience).
# In production, environment variables should be set by the host.
try:
    from dotenv import load_dotenv
    # Look for .env in the project root (one level above backend/)
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on real environment variables

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project (one level above this file's directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DATASET_PATH: str = os.path.join(_ROOT, "dataset", "spam.csv")

MODEL_DIR: str = os.path.join(_HERE, "model")
MODEL_PATH: str = os.path.join(MODEL_DIR, "model.pkl")
VOCABULARY_PATH: str = os.path.join(MODEL_DIR, "vocabulary.pkl")

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

# Laplace smoothing parameter (alpha = 1 → add-one smoothing)
ALPHA: float = 1.0

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH: int = 10_000   # characters; enforced by API and CLI

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

FLASK_HOST: str = "127.0.0.1"
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = False          # Never True in production

# Flask session secret key — loaded from environment.
# Must be a long random string.  Never hard-code this.
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "")

# ---------------------------------------------------------------------------
# Google OAuth 2.0
# ---------------------------------------------------------------------------

# Path to the credentials.json downloaded from Google Cloud Console.
# Place it at the project root.  It is listed in .gitignore.
GOOGLE_CREDENTIALS_FILE: str = os.path.join(
    _ROOT, "credentials.json"
)

# Minimum required scope — read-only access to the user's Gmail messages.
# We never request write, send, delete, or full mail access.
GOOGLE_SCOPES: list = [
    "https://www.googleapis.com/auth/gmail.readonly",
    # openid + profile needed to read the user's email address for display
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# Redirect URI — must exactly match the URI configured in Google Cloud.
# For local development: http://localhost:5000/oauth2callback
OAUTH_REDIRECT_URI: str = os.environ.get(
    "OAUTH_REDIRECT_URI",
    "http://localhost:5000/oauth2callback",
)

# ---------------------------------------------------------------------------
# Gmail analysis limits
# ---------------------------------------------------------------------------

# Maximum emails fetched and classified per request.
# Hard-capped in gmail_service.py at 50 regardless of this value.
GMAIL_MAX_RESULTS: int = int(os.environ.get("GMAIL_MAX_RESULTS", "10"))
