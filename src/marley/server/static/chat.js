/**
 * MARley Chat UI — Production chat interface.
 *
 * Handles user input, API calls to /api/chat, and message rendering
 * with source references and abstention display.
 */

(function () {
    "use strict";

    const chatLog = document.getElementById("chat-log");
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query");
    const sendBtn = document.getElementById("send-btn");

    // --- API ---

    async function sendChat(query) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            const detail = await response.json().catch(() => null);
            throw new Error(detail?.detail || `Request failed (${response.status})`);
        }

        return response.json();
    }

    // --- Rendering ---

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function confidenceClass(score) {
        if (score >= 0.6) return "confidence-high";
        if (score >= 0.3) return "confidence-medium";
        return "confidence-low";
    }

    function renderUserMessage(text) {
        const msg = document.createElement("div");
        msg.className = "message user";
        msg.innerHTML =
            '<div class="message-label">You</div>' +
            '<div class="message-bubble">' + escapeHtml(text) + '</div>';
        chatLog.appendChild(msg);
    }

    function renderLoadingMessage() {
        const msg = document.createElement("div");
        msg.className = "message system";
        msg.id = "loading-msg";
        msg.innerHTML =
            '<div class="message-label">MARley</div>' +
            '<div class="loading-message"><span class="spinner"></span> Generating answer&hellip;</div>';
        chatLog.appendChild(msg);
        scrollToBottom();
    }

    function removeLoadingMessage() {
        const el = document.getElementById("loading-msg");
        if (el) el.remove();
    }

    function renderSystemMessage(data) {
        const msg = document.createElement("div");
        msg.className = "message system";

        let html = '<div class="message-label">MARley</div>';

        if (data.abstained) {
            html += renderAbstention(data);
        } else {
            html += '<div class="message-bubble">' + escapeHtml(data.answer) + '</div>';
        }

        // Confidence badge
        const cls = confidenceClass(data.confidence);
        html += '<span class="confidence-badge ' + cls + '">Confidence: ' +
            (data.confidence * 100).toFixed(1) + '%</span>';

        // Sources
        if (data.sources && data.sources.length > 0) {
            html += renderSources(data.sources);
        }

        msg.innerHTML = html;
        chatLog.appendChild(msg);

        // Attach toggle listener
        const toggle = msg.querySelector(".sources-toggle");
        if (toggle) {
            toggle.addEventListener("click", function () {
                const list = msg.querySelector(".sources-list");
                list.classList.toggle("open");
                toggle.textContent = list.classList.contains("open")
                    ? "Hide sources" : "Show sources (" + data.sources.length + ")";
            });
        }
    }

    function renderAbstention(data) {
        const level = data.abstention_level || "?";
        const reason = data.abstention_reason || "No sufficient information found.";
        return '<div class="abstention-box">' +
            '<div class="abstention-title">Unable to Answer (Level ' + level + ')</div>' +
            '<div>' + escapeHtml(reason) + '</div>' +
            '<div class="abstention-hint">' +
                'For questions not covered by the study regulations, please contact the ' +
                'Academic Advisory Office (Studienberatung) directly.' +
            '</div>' +
            '</div>';
    }

    function renderSources(sources) {
        let html = '<button class="sources-toggle">Show sources (' + sources.length + ')</button>';
        html += '<div class="sources-list">';
        for (const s of sources) {
            html += '<div class="source-item">' +
                '<span class="source-id">' + escapeHtml(s.chunk_id) + '</span>' +
                '<span class="source-score">' + s.score.toFixed(4) + '</span>' +
                '<div class="source-text">' + escapeHtml(s.text.substring(0, 300)) + '</div>' +
                '</div>';
        }
        html += '</div>';
        return html;
    }

    function renderError(message) {
        const msg = document.createElement("div");
        msg.className = "message system";
        msg.innerHTML =
            '<div class="message-label">MARley</div>' +
            '<div class="message-bubble" style="border-color: var(--color-error); color: var(--color-error);">' +
            escapeHtml(message) + '</div>';
        chatLog.appendChild(msg);
    }

    function scrollToBottom() {
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function setLoading(loading) {
        queryInput.disabled = loading;
        sendBtn.disabled = loading;
        if (!loading) queryInput.focus();
    }

    // --- Auto-resize textarea ---

    queryInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });

    // --- Submit handler ---

    chatForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        renderUserMessage(query);
        queryInput.value = "";
        queryInput.style.height = "auto";
        scrollToBottom();
        setLoading(true);
        renderLoadingMessage();

        try {
            const data = await sendChat(query);
            removeLoadingMessage();
            renderSystemMessage(data);
        } catch (err) {
            removeLoadingMessage();
            renderError("Error: " + err.message);
        }

        setLoading(false);
        scrollToBottom();
    });

    // Submit on Enter (Shift+Enter for newline)
    queryInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });
})();
