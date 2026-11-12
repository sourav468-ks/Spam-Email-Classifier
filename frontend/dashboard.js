/**
 * dashboard.js — Frontend logic for the SpamGuard Gmail Dashboard.
 *
 * Responsibilities:
 *   - Verify authentication status on load (redirect if not authed)
 *   - Display user email in header
 *   - Handle "Analyze" button → GET /gmail/analyze
 *   - Render results table with SPAM/HAM badges
 *   - Handle email detail modal
 *   - Handle logout
 *   - Handle error states (401 → redirect to login, network errors, etc.)
 *
 * Security notes:
 *   - ALL email content (subject, sender, date) is set via textContent.
 *     NEVER via innerHTML. This prevents XSS regardless of email content.
 *   - Email body text is NOT displayed by default (not returned by API).
 *   - OAuth tokens are NEVER touched by this script.
 *   - Error messages are rendered via textContent.
 */

"use strict";

/* -------------------------------------------------------------------------- */
/* DOM references                                                              */
/* -------------------------------------------------------------------------- */
const userEmailEl     = document.getElementById("user-email");
const analyzeBtn      = document.getElementById("analyze-btn");
const analyzeText     = document.getElementById("analyze-text");
const analyzeSpinner  = document.getElementById("analyze-spinner");
const analyzeIcon     = document.getElementById("analyze-icon");
const gmailSearch     = document.getElementById("gmail-search");
const maxResultsSel   = document.getElementById("max-results-select");
const dashboardError  = document.getElementById("dashboard-error");
const dashboardErrTxt = document.getElementById("dashboard-error-text");
const resultsSection  = document.getElementById("results-section");
const resultsCount    = document.getElementById("results-count");
const tableBody       = document.getElementById("gmail-table-body");
const emptyState      = document.getElementById("empty-state");
const initialState    = document.getElementById("initial-state");
const statsGrid       = document.getElementById("stats-grid");
const statTotal       = document.getElementById("stat-total");
const statSpam        = document.getElementById("stat-spam");
const statHam         = document.getElementById("stat-ham");
const statRate        = document.getElementById("stat-rate");

// Modal
const modalOverlay    = document.getElementById("modal-overlay");
const modalClose      = document.getElementById("modal-close");
const modalBadge      = document.getElementById("modal-badge");
const modalSubject    = document.getElementById("modal-subject");
const modalSender     = document.getElementById("modal-sender");
const modalDate       = document.getElementById("modal-date");
const modalSpamFill   = document.getElementById("modal-spam-fill");
const modalHamFill    = document.getElementById("modal-ham-fill");
const modalSpamVal    = document.getElementById("modal-spam-val");
const modalHamVal     = document.getElementById("modal-ham-val");

/* -------------------------------------------------------------------------- */
/* Initialization                                                              */
/* -------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", async () => {
    await checkAuthStatus();

    analyzeBtn.addEventListener("click", handleAnalyze);

    // Allow Enter key in search box to trigger analysis
    gmailSearch.addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleAnalyze();
    });

    // Modal close
    modalClose.addEventListener("click", closeModal);
    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeModal();
    });
});

/* -------------------------------------------------------------------------- */
/* Auth check                                                                  */
/* -------------------------------------------------------------------------- */
async function checkAuthStatus() {
    try {
        const res = await fetch("/auth/status");
        if (!res.ok) throw new Error("Auth status check failed");
        const data = await res.json();

        if (!data.authenticated) {
            // Not logged in — redirect to home
            window.location.href = "/?error=auth_required";
            return;
        }

        // Display user info — textContent prevents XSS
        const displayName = data.name || data.email || "Google User";
        userEmailEl.textContent = displayName;

    } catch (e) {
        // Network error — still try to stay on page but show warning
        userEmailEl.textContent = "Connected";
    }
}

/* -------------------------------------------------------------------------- */
/* Analyze handler                                                             */
/* -------------------------------------------------------------------------- */
async function handleAnalyze() {
    const query      = gmailSearch.value.trim();
    const maxResults = maxResultsSel.value;

    hideDashboardError();
    setLoading(true);

    try {
        const params = new URLSearchParams({
            max_results: maxResults,
        });
        if (query) params.append("query", query);

        const res = await fetch(`/gmail/analyze?${params.toString()}`);
        const data = await res.json();

        if (!res.ok) {
            // Handle specific error codes
            if (res.status === 401) {
                // Session expired or revoked → redirect to home
                window.location.href = "/?error=auth_required";
                return;
            }
            showDashboardError(
                data.error || "Failed to analyze Gmail. Please try again."
            );
            return;
        }

        renderResults(data.results || []);

    } catch (networkErr) {
        showDashboardError(
            "Could not reach the server. Make sure the backend is running."
        );
    } finally {
        setLoading(false);
    }
}

/* -------------------------------------------------------------------------- */
/* Results rendering                                                           */
/* -------------------------------------------------------------------------- */
function renderResults(results) {
    // Clear previous results
    tableBody.innerHTML = "";
    initialState.hidden = true;
    emptyState.hidden   = true;

    if (!results || results.length === 0) {
        emptyState.hidden   = false;
        statsGrid.hidden    = true;
        resultsCount.textContent = "";
        return;
    }

    // Compute stats
    const spamCount = results.filter(r => r.prediction === "spam").length;
    const hamCount  = results.length - spamCount;
    const spamRate  = results.length > 0
        ? Math.round((spamCount / results.length) * 100)
        : 0;

    // Update summary
    resultsCount.textContent =
        `${results.length} email${results.length !== 1 ? "s" : ""} analyzed`;

    // Update stats cards
    statTotal.textContent = results.length;
    statSpam.textContent  = spamCount;
    statHam.textContent   = hamCount;
    statRate.textContent  = `${spamRate}%`;
    statsGrid.hidden      = false;

    // Build table rows — ALL content via textContent (XSS-safe)
    results.forEach((email, idx) => {
        const tr = document.createElement("tr");
        tr.className = "gmail-row";
        tr.setAttribute("data-index", idx);
        tr.setAttribute("tabindex", "0");
        tr.setAttribute("role", "button");
        tr.setAttribute("aria-label", `View email: ${email.subject}`);

        // Subject cell
        const tdSubject = document.createElement("td");
        tdSubject.className = "gmail-cell gmail-cell--subject";
        const subjectSpan = document.createElement("span");
        subjectSpan.className = "subject-text";
        // textContent — NEVER innerHTML — prevents XSS from email subject
        subjectSpan.textContent = truncate(email.subject, 60);
        tdSubject.appendChild(subjectSpan);

        // Sender cell
        const tdSender = document.createElement("td");
        tdSender.className = "gmail-cell gmail-cell--sender";
        // textContent — safe
        tdSender.textContent = truncate(email.sender, 40);

        // Date cell
        const tdDate = document.createElement("td");
        tdDate.className = "gmail-cell gmail-cell--date";
        tdDate.textContent = formatDate(email.date);

        // Result cell
        const tdResult = document.createElement("td");
        tdResult.className = "gmail-cell gmail-cell--result";
        const badge = document.createElement("span");
        badge.className = email.prediction === "spam"
            ? "gmail-badge gmail-badge--spam"
            : "gmail-badge gmail-badge--ham";
        // textContent — badge label is our own string, not email content
        badge.textContent = email.prediction === "spam" ? "🚨 SPAM" : "✅ HAM";
        tdResult.appendChild(badge);

        tr.appendChild(tdSubject);
        tr.appendChild(tdSender);
        tr.appendChild(tdDate);
        tr.appendChild(tdResult);

        // Click/keyboard handler to show detail modal
        tr.addEventListener("click", () => openModal(email));
        tr.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") openModal(email);
        });

        tableBody.appendChild(tr);
    });
}

/* -------------------------------------------------------------------------- */
/* Modal                                                                       */
/* -------------------------------------------------------------------------- */
function openModal(email) {
    const isSpam = email.prediction === "spam";

    // Badge — textContent
    modalBadge.className = isSpam
        ? "result-badge result-badge--spam"
        : "result-badge result-badge--ham";
    modalBadge.textContent = isSpam ? "🚨 SPAM" : "✅ HAM";

    // Metadata — textContent prevents XSS from any email field
    modalSubject.textContent = email.subject || "(No subject)";
    modalSender.textContent  = email.sender  || "(Unknown sender)";
    modalDate.textContent    = email.date    || "(Unknown date)";

    // Score bars
    renderModalScoreBars(email.score_spam, email.score_ham);

    modalOverlay.hidden = false;
    modalOverlay.removeAttribute("hidden");
    modalClose.focus();
    document.body.style.overflow = "hidden";
}

function closeModal() {
    modalOverlay.hidden = true;
    modalOverlay.setAttribute("hidden", "");
    document.body.style.overflow = "";
    analyzeBtn.focus();
}

function renderModalScoreBars(scoreSpam, scoreHam) {
    if (scoreSpam === null || scoreHam === null) {
        modalSpamFill.style.width = "0%";
        modalHamFill.style.width  = "0%";
        modalSpamVal.textContent  = "—";
        modalHamVal.textContent   = "—";
        return;
    }
    const minScore   = Math.min(scoreSpam, scoreHam);
    const spamShifted = scoreSpam - minScore;
    const hamShifted  = scoreHam  - minScore;
    const total       = spamShifted + hamShifted || 1;

    modalSpamFill.style.width = `${Math.round((spamShifted / total) * 100)}%`;
    modalHamFill.style.width  = `${Math.round((hamShifted  / total) * 100)}%`;
    // textContent for score numbers — these are numbers we control
    modalSpamVal.textContent  = scoreSpam.toFixed(2);
    modalHamVal.textContent   = scoreHam.toFixed(2);
}

/* -------------------------------------------------------------------------- */
/* Loading state                                                               */
/* -------------------------------------------------------------------------- */
function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
        analyzeBtn.classList.add("loading");
    } else {
        analyzeBtn.classList.remove("loading");
    }
}

/* -------------------------------------------------------------------------- */
/* Error display                                                               */
/* -------------------------------------------------------------------------- */
function showDashboardError(msg) {
    // textContent — never innerHTML
    dashboardErrTxt.textContent = msg;
    dashboardError.classList.add("visible");
}

function hideDashboardError() {
    dashboardError.classList.remove("visible");
}

/* -------------------------------------------------------------------------- */
/* Utilities                                                                   */
/* -------------------------------------------------------------------------- */
function truncate(str, maxLen) {
    if (!str) return "";
    return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

function formatDate(dateStr) {
    if (!dateStr || dateStr === "(Unknown date)") return dateStr || "—";
    try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        return d.toLocaleDateString(undefined, {
            month: "short",
            day:   "numeric",
            year:  "numeric",
        });
    } catch {
        return dateStr;
    }
}
