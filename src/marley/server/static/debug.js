/**
 * MARley Debug UI — Extended interface with configuration panel.
 *
 * Provides full control over pipeline configuration and displays
 * detailed retrieval results including chunk scores and metadata.
 */

(function () {
    "use strict";

    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query");
    const sendBtn = document.getElementById("send-btn");
    const resultDiv = document.getElementById("result");
    const serverStatus = document.getElementById("server-status");

    // Config elements
    const retrieverType = document.getElementById("retriever-type");
    const kbCheckboxes = document.getElementById("kb-checkboxes");
    const strategy = document.getElementById("strategy");
    const kValue = document.getElementById("k-value");
    const thresholdInput = document.getElementById("threshold");

    // --- API ---

    async function fetchOptions() {
        try {
            const resp = await fetch("/api/options");
            if (!resp.ok) throw new Error("Failed to load options");
            return resp.json();
        } catch (err) {
            return null;
        }
    }

    async function fetchHealth() {
        try {
            const resp = await fetch("/api/health");
            if (!resp.ok) throw new Error("Health check failed");
            return resp.json();
        } catch (err) {
            return null;
        }
    }

    async function sendQuery(query, config) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, ...config }),
        });

        if (!response.ok) {
            const detail = await response.json().catch(() => null);
            throw new Error(detail?.detail || "Request failed (" + response.status + ")");
        }

        return response.json();
    }

    // --- Config ---

    function getConfig() {
        const kbs = [];
        kbCheckboxes.querySelectorAll("input:checked").forEach(function (cb) {
            kbs.push(cb.value);
        });

        const config = {
            retriever_type: retrieverType.value,
            knowledge_bases: kbs,
            strategy: strategy.value,
            k: parseInt(kValue.value, 10) || 5,
        };

        const t = thresholdInput.value.trim();
        if (t !== "") {
            config.threshold = parseFloat(t);
        }

        return config;
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

    function renderResult(data) {
        let html = "";

        // Config summary
        html += '<div class="config-summary">';
        html += '<span class="config-tag">' + data.config.retriever_type + '</span>';
        html += '<span class="config-tag">' + data.config.strategy + '</span>';
        html += '<span class="config-tag">k=' + data.config.k + '</span>';
        html += '<span class="config-tag">threshold=' + data.config.threshold + '</span>';
        html += '<span class="config-tag">' + data.config.normalization_strategy + '</span>';
        html += '<span class="config-tag">' + data.config.model + '</span>';
        html += '</div>';

        // Answer card
        html += '<div class="result-card">';
        html += '<h4>Answer</h4>';

        if (data.abstained) {
            html += '<div class="abstention-box">';
            html += '<div class="abstention-title">Abstained (Level ' +
                (data.abstention_level || "?") + ')</div>';
            html += '<div>' + escapeHtml(data.abstention_reason) + '</div>';
            html += '</div>';
        } else {
            html += '<div class="answer-text">' + escapeHtml(data.answer) + '</div>';
        }

        // Confidence
        const cls = confidenceClass(data.confidence);
        html += '<div style="margin-top: 10px;">';
        html += '<span class="confidence-badge ' + cls + '">Confidence: ' +
            (data.confidence * 100).toFixed(1) + '%</span>';
        html += ' <span style="font-size: 12px; color: var(--color-text-muted);">(' +
            data.confidence.toFixed(4) + ')</span>';
        html += '</div>';
        html += '</div>';

        // Sources table
        if (data.sources && data.sources.length > 0) {
            html += '<div class="result-card">';
            html += '<h4>Retrieved Chunks (' + data.sources.length + ')</h4>';
            html += renderChunkTable(data.sources);
            html += '</div>';
        }

        resultDiv.innerHTML = html;
    }

    function renderChunkTable(sources) {
        let html = '<table class="chunks-table">';
        html += '<thead><tr>' +
            '<th>#</th><th>Chunk ID</th><th>Score</th><th>Text Preview</th>' +
            '</tr></thead><tbody>';

        sources.forEach(function (s, i) {
            html += '<tr>';
            html += '<td>' + (i + 1) + '</td>';
            html += '<td class="chunk-id">' + escapeHtml(s.chunk_id) + '</td>';
            html += '<td class="chunk-score">' + s.score.toFixed(4) + '</td>';
            html += '<td class="chunk-text">' + escapeHtml(s.text.substring(0, 200)) + '</td>';
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    }

    function renderError(message) {
        resultDiv.innerHTML =
            '<div class="result-card" style="border-color: var(--color-error);">' +
            '<h4 style="color: var(--color-error);">Error</h4>' +
            '<div>' + escapeHtml(message) + '</div>' +
            '</div>';
    }

    function renderLoading() {
        resultDiv.innerHTML =
            '<div class="loading-message"><span class="spinner"></span> Processing query&hellip;</div>';
    }

    function setLoading(loading) {
        queryInput.disabled = loading;
        sendBtn.disabled = loading;
        if (!loading) queryInput.focus();
    }

    // --- Server status ---

    async function updateServerStatus() {
        const health = await fetchHealth();
        if (!health) {
            serverStatus.innerHTML =
                '<span class="status-dot unavailable"></span> Server unreachable';
            return;
        }

        const dotClass = health.ollama === "connected" ? "connected" : "unavailable";
        serverStatus.innerHTML =
            '<span class="status-dot ' + dotClass + '"></span> ' +
            'Ollama: ' + health.ollama + '<br>' +
            'Model: ' + health.model + '<br>' +
            'Cached retrievers: ' + health.cached_retrievers + '<br>' +
            'KBs: ' + (health.knowledge_bases.join(", ") || "none");
    }

    // --- Submit handler ---

    queryForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        setLoading(true);
        renderLoading();

        try {
            const config = getConfig();
            const data = await sendQuery(query, config);
            renderResult(data);
        } catch (err) {
            renderError(err.message);
        }

        setLoading(false);
        updateServerStatus();
    });

    // --- Init ---

    updateServerStatus();
})();
