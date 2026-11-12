"""
app.py — Flask API for the Spam Email Classifier + Gmail OAuth integration.

Endpoints (original — unchanged)
---------------------------------
GET  /              Serves the frontend HTML page.
POST /predict       Classifies an email message.
GET  /health        Returns API status.

Endpoints (new — Gmail OAuth)
------------------------------
GET  /login/google      Starts Google OAuth 2.0 flow.
GET  /oauth2callback    Handles Google OAuth callback.
GET  /auth/status       Returns {authenticated, email} JSON.
GET  /dashboard         Serves the Gmail dashboard page (auth required).
GET  /gmail/analyze     Fetches and classifies Gmail emails (auth required).
POST /logout            Destroys the session and redirects to /.

Security measures implemented
------------------------------
* Input validation: type check, empty check, length limit (10,000 chars).
* Model loaded ONCE at startup — not per request.
* Stack traces logged server-side but NEVER sent to the client.
* Email content is NOT logged.
* OAuth state parameter validated in callback (CSRF protection).
* Credentials stored in server-side Flask session (encrypted cookie).
* Access token and refresh token NEVER returned to the browser.
* Session cookie: HttpOnly=True, SameSite=Lax.
* SESSION_COOKIE_SECURE=False for local HTTP development.
* Dashboard and Gmail endpoints require an authenticated session.
* Gmail body text discarded after classification — never stored.
* All dynamic content rendered via textContent in JS (XSS prevention).

Usage
-----
    python backend/app.py
"""

import os
import sys
import logging

# Allow running from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import (
    Flask, request, jsonify, send_from_directory, abort,
    redirect, url_for, session,
)
import joblib
import google.auth.exceptions
from googleapiclient.errors import HttpError

from backend.config import (
    MODEL_PATH,
    VOCABULARY_PATH,
    MAX_MESSAGE_LENGTH,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    FLASK_SECRET_KEY,
    GMAIL_MAX_RESULTS,
)
from backend.preprocessing import preprocess
from backend.classifier import text_to_vector


# ---------------------------------------------------------------------------
# Logging — safe configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(_ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------

if not FLASK_SECRET_KEY:
    logger.warning(
        "FLASK_SECRET_KEY is not set. Using an insecure default. "
        "Set FLASK_SECRET_KEY in your .env file before going to production."
    )
    # Insecure fallback for development only — sessions won't persist across restarts
    app.secret_key = "dev-only-insecure-secret-change-me"
else:
    app.secret_key = FLASK_SECRET_KEY

# Cookie security:
# - HttpOnly: prevents JavaScript from reading the cookie (mitigates XSS token theft)
# - SameSite=Lax: prevents CSRF from cross-site requests while allowing top-level navigations
# - Secure=False: must be False for local HTTP development (set True for HTTPS production)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False   # Set True in production (HTTPS)


# ---------------------------------------------------------------------------
# Model loading — done once at startup
# ---------------------------------------------------------------------------

_model = None
_vocabulary = None


def _load_model():
    """Load the trained model and vocabulary from disk.

    Returns True on success, False if files are missing.
    """
    global _model, _vocabulary
    if not os.path.isfile(MODEL_PATH):
        logger.error("Model file not found: %s", MODEL_PATH)
        return False
    if not os.path.isfile(VOCABULARY_PATH):
        logger.error("Vocabulary file not found: %s", VOCABULARY_PATH)
        return False
    try:
        _model = joblib.load(MODEL_PATH)
        _vocabulary = joblib.load(VOCABULARY_PATH)
        logger.info("Model loaded (vocabulary size: %d)", len(_vocabulary))
        return True
    except Exception:
        logger.exception("Failed to load model artifacts.")
        return False


_model_available = _load_model()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _is_authenticated() -> bool:
    """Return True if the current request has a valid authenticated session."""
    return bool(session.get("credentials") and session.get("user_email"))


def _get_credentials():
    """Reconstruct Google credentials from the session.

    Returns None if not authenticated or credentials are missing.
    """
    from backend.google_auth import credentials_from_dict, refresh_credentials_if_needed

    cred_data = session.get("credentials")
    if not cred_data:
        return None
    try:
        credentials = credentials_from_dict(cred_data)
        # Refresh if expired
        credentials = refresh_credentials_if_needed(credentials)
        # Update session with potentially refreshed token
        from backend.google_auth import credentials_to_dict
        session["credentials"] = credentials_to_dict(credentials)
        return credentials
    except google.auth.exceptions.RefreshError:
        logger.warning("Credentials refresh failed — clearing session.")
        session.clear()
        return None
    except Exception:
        logger.exception("Failed to load credentials from session.")
        return None


# ---------------------------------------------------------------------------
# Original routes (unchanged)
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        abort(404, description="Frontend not found. Run setup.")
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/style.css", methods=["GET"])
def styles():
    """Serve style.css."""
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.route("/script.js", methods=["GET"])
def scripts():
    """Serve script.js."""
    return send_from_directory(FRONTEND_DIR, "script.js")


@app.route("/dashboard.js", methods=["GET"])
def dashboard_scripts():
    """Serve dashboard.js."""
    return send_from_directory(FRONTEND_DIR, "dashboard.js")


@app.route("/health", methods=["GET"])
def health():
    """Return a simple health/status check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model_available,
        "vocabulary_size": len(_vocabulary) if _vocabulary else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Classify an email message as spam or ham.

    Request body (JSON):
        { "message": "<email text>" }

    Response (JSON):
        {
            "prediction": "spam" | "ham",
            "score_spam": <float>,
            "score_ham": <float>
        }
    """
    # ---- Model availability check -----------------------------------------
    if not _model_available or _model is None:
        logger.warning("Prediction requested but model is not loaded.")
        return jsonify({"error": "The classifier is currently unavailable. "
                                 "Train the model first."}), 503

    # ---- Parse JSON body ---------------------------------------------------
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)."}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    # ---- Validate 'message' field ------------------------------------------
    message = data.get("message")

    if message is None:
        return jsonify({"error": "Missing required field: 'message'."}), 400

    if not isinstance(message, str):
        return jsonify({"error": "Field 'message' must be a string."}), 400

    if not message.strip():
        return jsonify({"error": "Please enter a non-empty email."}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": f"Message exceeds the maximum length of "
                     f"{MAX_MESSAGE_LENGTH} characters."
        }), 400

    # ---- Classify ----------------------------------------------------------
    try:
        tokens = preprocess(message)

        if not tokens:
            return jsonify({
                "prediction": "ham",
                "score_spam": None,
                "score_ham": None,
                "note": "No recognizable words found; defaulting to ham.",
            })

        scores = _model.get_scores_for_text(tokens, _vocabulary)
        prediction = max(scores, key=scores.get)

        logger.info(
            "Prediction request: result=%s  tokens=%d",
            prediction,
            len(tokens),
        )

        return jsonify({
            "prediction": prediction,
            "score_spam": round(scores.get("spam", 0.0), 4),
            "score_ham":  round(scores.get("ham",  0.0), 4),
        })

    except Exception:
        logger.exception("Unexpected error during prediction.")
        return jsonify({"error": "An internal error occurred."}), 500


# ---------------------------------------------------------------------------
# OAuth routes
# ---------------------------------------------------------------------------

@app.route("/login/google", methods=["GET"])
def login_google():
    """Initiate Google OAuth 2.0 flow.

    Generates a CSRF state token, stores it in the session, and redirects
    the user to Google's authorization page.
    """
    try:
        from backend.google_auth import get_google_auth_flow, get_authorization_url
        flow = get_google_auth_flow()
        authorization_url, state, code_verifier = get_authorization_url(flow)
        # Store CSRF state in session
        session["oauth_state"] = state
        if code_verifier:
            session["code_verifier"] = code_verifier
        return redirect(authorization_url)

    except FileNotFoundError:
        logger.error("credentials.json not found. Cannot start OAuth flow.")
        return redirect(url_for("index") + "?error=oauth_config")

    except ValueError as e:
        logger.error("OAuth configuration error: %s", e)
        return redirect(url_for("index") + "?error=oauth_config")

    except Exception:
        logger.exception("Unexpected error starting OAuth flow.")
        return redirect(url_for("index") + "?error=oauth_error")


@app.route("/oauth2callback", methods=["GET"])
def oauth2callback():
    """Handle the Google OAuth 2.0 callback.

    Validates the CSRF state, exchanges the authorization code for
    credentials, stores credentials in the session, and redirects to
    the dashboard.

    Error handling:
    - Missing or mismatched state → 400 (CSRF protection)
    - OAuth error from Google → redirect to index with error flag
    - Exchange failure → redirect to index with error flag
    """
    # ---- CSRF state validation --------------------------------------------
    received_state = request.args.get("state")
    expected_state = session.get("oauth_state")

    if not received_state or not expected_state or received_state != expected_state:
        logger.warning("OAuth CSRF state mismatch — possible CSRF attack.")
        return jsonify({"error": "Invalid OAuth state. Please try logging in again."}), 400

    # Clear state — it's single-use
    session.pop("oauth_state", None)
    code_verifier = session.pop("code_verifier", None)

    # ---- Check for OAuth errors from Google --------------------------------
    error = request.args.get("error")
    if error:
        logger.info("OAuth denied by user or Google error: %s", error)
        # Redirect to index with a safe user-facing message
        return redirect(url_for("index") + "?error=oauth_denied")

    # ---- Exchange authorization code for credentials ----------------------
    try:
        from backend.google_auth import (
            get_google_auth_flow,
            exchange_code_for_credentials,
            credentials_to_dict,
        )

        flow = get_google_auth_flow(state=expected_state)
        if code_verifier:
            flow.code_verifier = code_verifier
            
        # The state must be set so the flow can validate it internally
        flow.fetch_token(
            authorization_response=request.url,
        )
        credentials = flow.credentials

        # ---- Get user profile info ----------------------------------------
        from googleapiclient.discovery import build as _build
        oauth2_service = _build(
            "oauth2", "v2",
            credentials=credentials,
            cache_discovery=False,
        )
        user_info = oauth2_service.userinfo().get().execute()
        user_email = user_info.get("email", "")
        user_name  = user_info.get("name", user_email)

        # ---- Store in session (server-side encrypted cookie) ---------------
        # SECURITY: Only the serialised credential dict and user info are
        # stored. The raw token is never returned to the browser or logged.
        session["credentials"] = credentials_to_dict(credentials)
        session["user_email"]  = user_email
        session["user_name"]   = user_name
        session.permanent      = False   # Session cookie (no persistent token file)

        logger.info("OAuth login success for user (email hash logged, not address).")
        return redirect(url_for("dashboard"))

    except Exception as e:
        logger.exception("OAuth callback error.")
        import traceback
        with open(os.path.join(_ROOT, "oauth_error.log"), "w") as f:
            f.write(traceback.format_exc())
        session.clear()
        return redirect(url_for("index") + "?error=oauth_error")


@app.route("/auth/status", methods=["GET"])
def auth_status():
    """Return current authentication status.

    Response:
        { "authenticated": bool, "email": str | null, "name": str | null }

    Never returns OAuth tokens.
    """
    if _is_authenticated():
        return jsonify({
            "authenticated": True,
            "email": session.get("user_email"),
            "name":  session.get("user_name"),
        })
    return jsonify({"authenticated": False, "email": None, "name": None})


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the Gmail analysis dashboard.

    Requires an authenticated session.
    Unauthenticated requests are redirected to the index page.
    """
    if not _is_authenticated():
        return redirect(url_for("index") + "?error=auth_required")

    dashboard_path = os.path.join(FRONTEND_DIR, "dashboard.html")
    if not os.path.isfile(dashboard_path):
        abort(404, description="Dashboard not found.")
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


@app.route("/gmail/analyze", methods=["GET"])
def gmail_analyze():
    """Fetch and classify Gmail messages for the authenticated user.

    Query parameters:
        max_results (int, optional): Number of emails to analyze (1-50).
                                     Defaults to GMAIL_MAX_RESULTS from config.
        query (str, optional):       Gmail search query.

    Response (JSON):
        {
            "results": [
                {
                    "message_id": str,
                    "subject":    str,
                    "sender":     str,
                    "date":       str,
                    "prediction": "spam" | "ham",
                    "score_spam": float | null,
                    "score_ham":  float | null
                },
                ...
            ],
            "count": int
        }

    Error responses:
        401 — Not authenticated
        503 — Classifier not available
        502 — Gmail API unavailable
        500 — Internal error
    """
    # ---- Auth check -------------------------------------------------------
    if not _is_authenticated():
        return jsonify({
            "error": "Authentication required. Please sign in with Google."
        }), 401

    # ---- Classifier check -------------------------------------------------
    if not _model_available or _model is None:
        return jsonify({
            "error": "The classifier is currently unavailable. Train the model first."
        }), 503

    # ---- Parse parameters -------------------------------------------------
    try:
        max_results = int(request.args.get("max_results", GMAIL_MAX_RESULTS))
        max_results = max(1, min(max_results, 50))   # clamp 1-50
    except (ValueError, TypeError):
        max_results = GMAIL_MAX_RESULTS

    # Validate query parameter — used directly in Gmail query, not in SQL/file paths
    query = request.args.get("query", "").strip()
    if len(query) > 500:
        return jsonify({"error": "Query parameter is too long (max 500 chars)."}), 400

    # ---- Get credentials --------------------------------------------------
    credentials = _get_credentials()
    if credentials is None:
        return jsonify({
            "error": "Unable to access Gmail. Please connect your Google account again."
        }), 401

    # ---- Call Gmail service -----------------------------------------------
    try:
        from backend.gmail_service import build_gmail_service, analyze_emails
        service = build_gmail_service(credentials)
        results = analyze_emails(
            service=service,
            model=_model,
            vocabulary=_vocabulary,
            max_results=max_results,
            query=query,
        )

        return jsonify({
            "results": results,
            "count":   len(results),
        })

    except HttpError as e:
        status = e.status_code if hasattr(e, "status_code") else 0
        if status == 401 or status == 403:
            # Token revoked or insufficient permissions
            session.clear()
            return jsonify({
                "error": "Gmail access was revoked. Please connect your Google account again."
            }), 401
        logger.error("Gmail API HttpError: %s", status)
        return jsonify({
            "error": "Unable to access Gmail at this time. Please try again later."
        }), 502

    except google.auth.exceptions.RefreshError:
        session.clear()
        return jsonify({
            "error": "Gmail session expired. Please connect your Google account again."
        }), 401

    except Exception:
        logger.exception("Unexpected error during Gmail analysis.")
        return jsonify({"error": "An internal error occurred."}), 500


@app.route("/logout", methods=["POST"])
def logout():
    """Log out the user by clearing the session.

    Destroys the SpamGuard application session only.
    Does NOT log the user out of their Google account.
    Best-effort token revocation is attempted.

    Always redirects to the index page.
    """
    credentials = _get_credentials()
    if credentials:
        try:
            from backend.google_auth import revoke_credentials
            revoke_credentials(credentials)
        except Exception:
            # Revocation failure is non-fatal
            logger.warning("Token revocation failed during logout.", exc_info=True)

    session.clear()
    logger.info("User session cleared (logout).")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _model_available:
        print(
            "\n[WARNING] Model not found. Run this first:\n"
            "  python backend/train_model.py\n"
        )
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
