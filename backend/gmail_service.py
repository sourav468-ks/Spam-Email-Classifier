"""
gmail_service.py — Gmail API operations for SpamGuard.

This module fetches email messages, parses MIME content safely,
and runs them through the existing SpamGuard classifier.

Design constraints
------------------
* Gmail HTML is NEVER rendered or executed — we extract plain text only.
* Complete email bodies are NEVER logged.
* Access tokens are NEVER logged.
* Emails are not stored — they are fetched, classified, and discarded.
* Gmail API usage is rate-limited by GMAIL_MAX_RESULTS per request.

MIME handling
-------------
Emails can have the following structures:
  text/plain                       → use directly
  text/html                        → strip tags, use text
  multipart/alternative            → prefer text/plain part
  multipart/mixed                  → recurse into parts
  other MIME types                 → skip silently

Security
--------
* No email HTML is injected into the page (handled in dashboard.js too).
* Attachments are ignored.
* Malformed MIME does not crash the server.
"""

import base64
import logging
import re
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

from backend.preprocessing import preprocess
from backend.config import GMAIL_MAX_RESULTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gmail service factory
# ---------------------------------------------------------------------------

def build_gmail_service(credentials: Credentials):
    """Build and return a Gmail API service client.

    Parameters
    ----------
    credentials : google.oauth2.credentials.Credentials
        Valid (non-expired) credentials with gmail.readonly scope.

    Returns
    -------
    googleapiclient resource for Gmail v1.
    """
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


# ---------------------------------------------------------------------------
# Message ID fetching
# ---------------------------------------------------------------------------

def fetch_message_ids(
    service,
    max_results: int = GMAIL_MAX_RESULTS,
    query: str = "",
) -> list[str]:
    """Return a list of Gmail message IDs for the authenticated user.

    Fetches only IDs — the full message bodies are fetched separately
    so we don't download everything at once.

    Parameters
    ----------
    service : Gmail API resource
    max_results : int
        Maximum number of message IDs to return (default from config).
        Hard-capped at 50 to prevent runaway API usage.
    query : str
        Optional Gmail search query (e.g. "is:unread" or "from:boss@co.com").
        Uses Gmail's official query syntax — no local filtering.

    Returns
    -------
    list of str — Gmail message IDs.
    """
    # Hard cap to prevent runaway API usage
    safe_max = min(max_results, 50)

    try:
        kwargs = {
            "userId": "me",
            "maxResults": safe_max,
        }
        if query and query.strip():
            kwargs["q"] = query.strip()

        result = service.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])
        return [m["id"] for m in messages]

    except HttpError as e:
        logger.error("Gmail API error fetching message list: %s", e.status_code)
        raise


# ---------------------------------------------------------------------------
# Single message fetching
# ---------------------------------------------------------------------------

def fetch_message(service, message_id: str) -> dict:
    """Fetch a single Gmail message in full format.

    Parameters
    ----------
    service : Gmail API resource
    message_id : str
        Gmail message ID.

    Returns
    -------
    dict — raw Gmail message payload.
    """
    try:
        return service.users().messages().get(
            userId="me",
            id=message_id,
            format="full",
        ).execute()
    except HttpError as e:
        logger.error(
            "Gmail API error fetching message %s: %s",
            message_id[:8] + "...",   # partial ID only in logs
            e.status_code,
        )
        raise


# ---------------------------------------------------------------------------
# MIME parsing — safe plain-text extraction
# ---------------------------------------------------------------------------

def _decode_body_data(data: str) -> str:
    """Decode base64url-encoded Gmail body data to a UTF-8 string.

    Parameters
    ----------
    data : str
        Base64url-encoded string from Gmail API payload.

    Returns
    -------
    str — decoded text, empty string on failure.
    """
    if not data:
        return ""
    try:
        # Gmail uses URL-safe base64
        decoded_bytes = base64.urlsafe_b64decode(data + "==")
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to decode body data.", exc_info=True)
        return ""


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and return plain text.

    This is a deliberate best-effort strip — we do NOT parse or execute
    the HTML. We use a simple regex to remove tags, which is safe because
    we only need rough plain-text for classification purposes.

    Parameters
    ----------
    html : str
        Raw HTML content from an email.

    Returns
    -------
    str — text with HTML tags stripped.
    """
    # Remove script/style blocks entirely (content too)
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_plain_text_from_part(part: dict) -> Optional[str]:
    """Recursively extract plain text from a Gmail message part.

    Prefers text/plain over text/html.
    Skips attachments (parts with a filename).
    Returns None if no usable text is found.

    Parameters
    ----------
    part : dict
        A Gmail API message part.

    Returns
    -------
    str or None
    """
    mime_type = part.get("mimeType", "")
    body = part.get("body", {})
    parts = part.get("parts", [])

    # Skip attachments
    if part.get("filename"):
        return None

    # Leaf part: text/plain
    if mime_type == "text/plain":
        data = body.get("data", "")
        return _decode_body_data(data) if data else ""

    # Leaf part: text/html — strip tags for plain text
    if mime_type == "text/html":
        data = body.get("data", "")
        raw_html = _decode_body_data(data) if data else ""
        return _strip_html_tags(raw_html)

    # Multipart: recurse
    if mime_type.startswith("multipart/"):
        plain_candidate = None
        html_candidate = None

        for sub_part in parts:
            sub_mime = sub_part.get("mimeType", "")
            sub_body = sub_part.get("body", {})

            if sub_mime == "text/plain":
                data = sub_body.get("data", "")
                if data:
                    plain_candidate = _decode_body_data(data)

            elif sub_mime == "text/html":
                data = sub_body.get("data", "")
                if data:
                    raw_html = _decode_body_data(data)
                    html_candidate = _strip_html_tags(raw_html)

            elif sub_mime.startswith("multipart/"):
                # Recurse into nested multipart
                nested = _extract_plain_text_from_part(sub_part)
                if nested is not None and not plain_candidate:
                    plain_candidate = nested

        # Prefer plain text over HTML-derived text
        return plain_candidate if plain_candidate is not None else html_candidate

    # Unknown MIME type — skip silently
    return None


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------

def _get_header(headers: list, name: str) -> str:
    """Return the value of a named header from a Gmail headers list.

    Case-insensitive lookup. Returns empty string if not found.
    """
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


# ---------------------------------------------------------------------------
# Email data extraction
# ---------------------------------------------------------------------------

def extract_email_data(message: dict) -> dict:
    """Parse a raw Gmail message and return structured email data.

    Extracts:
      - message_id: str
      - subject: str
      - sender: str
      - date: str
      - body_text: str  (plain text for classification)

    The body_text is used only for classification and is NOT stored.

    Parameters
    ----------
    message : dict
        Raw Gmail API message from :func:`fetch_message`.

    Returns
    -------
    dict with keys: message_id, subject, sender, date, body_text.
    """
    message_id = message.get("id", "")
    payload = message.get("payload", {})
    headers = payload.get("headers", [])

    subject = _get_header(headers, "Subject") or "(No subject)"
    sender  = _get_header(headers, "From")    or "(Unknown sender)"
    date    = _get_header(headers, "Date")    or "(Unknown date)"

    # Try to extract plain text from the payload
    body_text = _extract_plain_text_from_part(payload)
    if body_text is None:
        body_text = ""

    return {
        "message_id": message_id,
        "subject":    subject,
        "sender":     sender,
        "date":       date,
        "body_text":  body_text,
    }


# ---------------------------------------------------------------------------
# End-to-end analysis pipeline
# ---------------------------------------------------------------------------

def analyze_emails(
    service,
    model,
    vocabulary: dict,
    max_results: int = GMAIL_MAX_RESULTS,
    query: str = "",
) -> list[dict]:
    """Fetch Gmail messages and classify each one using the SpamGuard model.

    This is the main integration point between Gmail and the classifier.
    Email bodies are NOT stored — they are fetched, classified, and discarded.

    Pipeline
    --------
    Gmail API → fetch N message IDs
        ↓
    fetch full message for each ID
        ↓
    extract_email_data() → subject, sender, date, body_text
        ↓
    preprocess(body_text) → tokens   [existing preprocessing.py]
        ↓
    model.get_scores_for_text(tokens, vocabulary)  [existing classifier]
        ↓
    Return list of result dicts (body_text NOT included in results)

    Parameters
    ----------
    service : Gmail API resource
        Authenticated service from :func:`build_gmail_service`.
    model : NaiveBayesClassifier
        Loaded SpamGuard model (from app.py globals).
    vocabulary : dict
        Loaded SpamGuard vocabulary (from app.py globals).
    max_results : int
        Maximum emails to fetch and classify (hard-capped at 50).
    query : str
        Optional Gmail search query.

    Returns
    -------
    list of dict, each with:
        message_id, subject, sender, date, prediction, score_spam, score_ham
    """
    safe_max = min(max_results, 50)
    results = []

    # Step 1: Fetch message IDs
    message_ids = fetch_message_ids(service, max_results=safe_max, query=query)

    if not message_ids:
        return []

    # Step 2: Fetch + classify each message
    for msg_id in message_ids:
        try:
            raw_message = fetch_message(service, msg_id)
            email_data = extract_email_data(raw_message)

            body_text = email_data.get("body_text", "")

            # Step 3: Preprocess using the existing preprocessing pipeline
            tokens = preprocess(body_text)

            # Step 4: Classify using the existing SpamGuard model
            if tokens:
                scores = model.get_scores_for_text(tokens, vocabulary)
                prediction = max(scores, key=scores.get)
                score_spam = round(scores.get("spam", 0.0), 4)
                score_ham  = round(scores.get("ham",  0.0), 4)
            else:
                # No tokens (empty/unreadable body) → default to ham
                prediction = "ham"
                score_spam = None
                score_ham  = None

            results.append({
                "message_id": email_data["message_id"],
                "subject":    email_data["subject"],
                "sender":     email_data["sender"],
                "date":       email_data["date"],
                "prediction": prediction,
                "score_spam": score_spam,
                "score_ham":  score_ham,
            })
            # NOTE: body_text is deliberately NOT included in results.
            # It is classified and discarded — never stored, never returned.

        except HttpError as e:
            # Log with minimal info (no email content)
            logger.warning(
                "Skipping message due to Gmail API error (status %s).",
                e.status_code,
            )
            continue
        except Exception:
            logger.warning(
                "Skipping message due to unexpected error.", exc_info=True
            )
            continue

    logger.info(
        "Analyzed %d email(s), query=%r",
        len(results),
        query[:50] if query else "",  # truncate long queries in logs
    )
    return results
