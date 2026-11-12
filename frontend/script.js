/**
 * script.js — Frontend logic for the Spam Email Classifier
 *
 * Responsibilities:
 *   - Handle the Classify button click
 *   - Validate input client-side (empty, too long)
 *   - Call POST /predict via fetch
 *   - Render the result (SPAM / HAM badge, score bars, pipeline steps)
 *   - Handle errors gracefully
 *
 * Security notes:
 *   - User-supplied email text is NEVER injected via innerHTML.
 *     All dynamic text uses textContent to prevent XSS.
 *   - Input is validated before sending to the server.
 *   - Error messages are escaped before display.
 */

"use strict";

/* -------------------------------------------------------------------------- */
/* Constants                                                                   */
/* -------------------------------------------------------------------------- */
const MAX_CHARS     = 10000;
const WARN_CHARS    = 8000;
const API_ENDPOINT  = "/predict";

/* Example messages for the quick-fill chips */
const EXAMPLES = [
    { label: "🚨 Spam", type: "spam", text: "CONGRATULATIONS! You have WON a FREE iPhone 15! Click here NOW to claim your prize before it expires!!!" },
    { label: "🚨 Spam", type: "spam", text: "URGENT: Your bank account has been suspended. Verify your details immediately at secure-banklogin.xyz" },
    { label: "🚨 Spam", type: "spam", text: "Win $1,000,000 cash prize! Limited time offer! Call 1-800-FREE-WIN now. No purchase necessary." },
    { label: "✅ Ham",  type: "ham",  text: "Hi Sarah, are you free for a quick call tomorrow at 2 PM to discuss the project timeline?" },
    { label: "✅ Ham",  type: "ham",  text: "Team, please review the attached agenda for Thursday's sprint planning meeting. Let me know if you have any additions." },
    { label: "✅ Ham",  type: "ham",  text: "Hey, just confirming our lunch reservation for Friday at noon. Looking forward to catching up!" },
];

/* Pipeline steps shown in the explainer */
const PIPELINE_STEPS = ["Raw Text", "Tokens", "Vocabulary", "BoW Vector", "Naive Bayes", "Spam / Ham"];

/* -------------------------------------------------------------------------- */
/* DOM references                                                              */
/* -------------------------------------------------------------------------- */
const emailTextarea   = document.getElementById("email-input");
const charCounter     = document.getElementById("char-counter");
const classifyBtn     = document.getElementById("classify-btn");
const btnText         = document.getElementById("btn-text");
const btnSpinner      = document.getElementById("btn-spinner");
const btnIcon         = document.getElementById("btn-icon");
const resultSection   = document.getElementById("result-section");
const resultBadge     = document.getElementById("result-badge");
const resultTitle     = document.getElementById("result-title");
const scoreSpamFill   = document.getElementById("score-spam-fill");
const scoreHamFill    = document.getElementById("score-ham-fill");
const scoreSpamVal    = document.getElementById("score-spam-val");
const scoreHamVal     = document.getElementById("score-ham-val");
const errorMessage    = document.getElementById("error-message");
const errorText       = document.getElementById("error-text");
const examplesChips   = document.getElementById("examples-chips");
const pipelineSteps   = document.getElementById("pipeline-steps");

/* -------------------------------------------------------------------------- */
/* Initialization                                                              */
/* -------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
    buildExampleChips();
    buildPipelineSteps();
    updateCharCounter();

    emailTextarea.addEventListener("input", () => {
        updateCharCounter();
        hideError();
        hideResult();
    });

    classifyBtn.addEventListener("click", handleClassify);

    // Allow Ctrl+Enter to submit
    emailTextarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            handleClassify();
        }
    });
});

/* -------------------------------------------------------------------------- */
/* Build example chips                                                         */
/* -------------------------------------------------------------------------- */
function buildExampleChips() {
    EXAMPLES.forEach((ex) => {
        const btn = document.createElement("button");
        btn.className = `example-chip example-chip--${ex.type}`;
        // textContent is safe — no HTML injection
        btn.textContent = ex.label;
        btn.setAttribute("title", ex.text.substring(0, 80) + "…");
        btn.addEventListener("click", () => {
            emailTextarea.value = ex.text;
            updateCharCounter();
            hideError();
            hideResult();
            emailTextarea.focus();
        });
        examplesChips.appendChild(btn);
    });
}

/* -------------------------------------------------------------------------- */
/* Build pipeline step indicators                                              */
/* -------------------------------------------------------------------------- */
function buildPipelineSteps() {
    PIPELINE_STEPS.forEach((step, i) => {
        const span = document.createElement("span");
        span.className = "pipeline-step";
        span.id = `pipeline-step-${i}`;
        span.textContent = step;
        pipelineSteps.appendChild(span);

        if (i < PIPELINE_STEPS.length - 1) {
            const arrow = document.createElement("span");
            arrow.className = "pipeline-arrow";
            arrow.setAttribute("aria-hidden", "true");
            arrow.textContent = "→";
            pipelineSteps.appendChild(arrow);
        }
    });
}

/* -------------------------------------------------------------------------- */
/* Char counter                                                                */
/* -------------------------------------------------------------------------- */
function updateCharCounter() {
    const len = emailTextarea.value.length;
    // textContent is safe here — len is a number
    charCounter.textContent = `${len} / ${MAX_CHARS}`;
    charCounter.classList.remove("warn", "error");
    if (len > MAX_CHARS) {
        charCounter.classList.add("error");
    } else if (len > WARN_CHARS) {
        charCounter.classList.add("warn");
    }
}

/* -------------------------------------------------------------------------- */
/* UI state helpers                                                            */
/* -------------------------------------------------------------------------- */
function setLoading(isLoading) {
    classifyBtn.disabled = isLoading;
    if (isLoading) {
        classifyBtn.classList.add("loading");
        animatePipeline();
    } else {
        classifyBtn.classList.remove("loading");
        resetPipelineAnimation();
    }
}

function showError(msg) {
    // Use textContent — never innerHTML — to prevent XSS
    errorText.textContent = msg;
    errorMessage.classList.add("visible");
    resultSection.classList.remove("visible");
}

function hideError() {
    errorMessage.classList.remove("visible");
}

function hideResult() {
    resultSection.classList.remove("visible");
}

/* -------------------------------------------------------------------------- */
/* Pipeline animation                                                          */
/* -------------------------------------------------------------------------- */
let _pipelineTimers = [];

function animatePipeline() {
    resetPipelineAnimation();
    PIPELINE_STEPS.forEach((_, i) => {
        const t = setTimeout(() => {
            const el = document.getElementById(`pipeline-step-${i}`);
            if (el) el.classList.add("active");
        }, i * 220);
        _pipelineTimers.push(t);
    });
}

function resetPipelineAnimation() {
    _pipelineTimers.forEach(clearTimeout);
    _pipelineTimers = [];
    PIPELINE_STEPS.forEach((_, i) => {
        const el = document.getElementById(`pipeline-step-${i}`);
        if (el) el.classList.remove("active");
    });
}

/* -------------------------------------------------------------------------- */
/* Score bar rendering                                                         */
/* -------------------------------------------------------------------------- */
function renderScoreBars(scoreSpam, scoreHam) {
    if (scoreSpam === null || scoreHam === null) {
        scoreSpamFill.style.width = "0%";
        scoreHamFill.style.width  = "0%";
        // textContent is safe — these are numbers/strings we control
        scoreSpamVal.textContent = "—";
        scoreHamVal.textContent  = "—";
        return;
    }

    // Both scores are negative log values.
    // Convert to a 0–100 relative display using softmax-like normalization.
    const minScore = Math.min(scoreSpam, scoreHam);
    const spamShifted = scoreSpam - minScore;
    const hamShifted  = scoreHam  - minScore;
    const total = spamShifted + hamShifted || 1;

    const spamPct = Math.round((spamShifted / total) * 100);
    const hamPct  = Math.round((hamShifted  / total) * 100);

    scoreSpamFill.style.width = `${spamPct}%`;
    scoreHamFill.style.width  = `${hamPct}%`;

    // Display raw log scores — NOT labeled as probabilities (per TRD §23)
    scoreSpamVal.textContent = scoreSpam.toFixed(2);
    scoreHamVal.textContent  = scoreHam.toFixed(2);
}

/* -------------------------------------------------------------------------- */
/* Result rendering                                                            */
/* -------------------------------------------------------------------------- */
function showResult(prediction, scoreSpam, scoreHam) {
    const isSpam = prediction === "spam";

    // Remove old class variants
    resultSection.classList.remove("result--spam", "result--ham");
    resultBadge.classList.remove("result-badge--spam", "result-badge--ham");
    resultTitle.classList.remove("result-title--spam", "result-title--ham");

    if (isSpam) {
        resultSection.classList.add("result--spam");
        resultBadge.classList.add("result-badge--spam");
        resultTitle.classList.add("result-title--spam");
        // textContent — safe, no HTML injection
        resultBadge.textContent = "🚨 SPAM";
        resultTitle.textContent = "This looks like spam.";
    } else {
        resultSection.classList.add("result--ham");
        resultBadge.classList.add("result-badge--ham");
        resultTitle.classList.add("result-title--ham");
        resultBadge.textContent = "✅ HAM";
        resultTitle.textContent = "This looks legitimate.";
    }

    renderScoreBars(scoreSpam, scoreHam);

    resultSection.classList.add("visible");
}

/* -------------------------------------------------------------------------- */
/* Main classify handler                                                       */
/* -------------------------------------------------------------------------- */
async function handleClassify() {
    const message = emailTextarea.value;

    // ---- Client-side validation ------------------------------------------
    if (!message.trim()) {
        showError("Please enter an email message before classifying.");
        emailTextarea.focus();
        return;
    }

    if (message.length > MAX_CHARS) {
        showError(`Message is too long. Maximum is ${MAX_CHARS} characters.`);
        return;
    }

    hideError();
    hideResult();
    setLoading(true);

    // ---- API call ---------------------------------------------------------
    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            // JSON.stringify handles escaping — no manual sanitization needed
            body: JSON.stringify({ message }),
        });

        const data = await response.json();

        if (!response.ok) {
            // Server returned an error — show safe message from server
            const errMsg = (data && typeof data.error === "string")
                ? data.error
                : "An error occurred. Please try again.";
            showError(errMsg);
            return;
        }

        if (!data.prediction) {
            showError("Unexpected response from server. Please try again.");
            return;
        }

        showResult(
            data.prediction,
            typeof data.score_spam === "number" ? data.score_spam : null,
            typeof data.score_ham  === "number" ? data.score_ham  : null,
        );

    } catch (networkError) {
        // Network failure — don't expose raw error to user
        showError(
            "Could not reach the classifier. " +
            "Make sure the server is running (python backend/app.py)."
        );
    } finally {
        setLoading(false);
    }
}
