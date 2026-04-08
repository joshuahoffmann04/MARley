/**
 * MARley Debug UI — Extended pipeline inspector.
 *
 * Provides full control over every pipeline parameter, shows the complete
 * retrieval result table with scores and page metadata, and integrates the
 * PDF slide-in viewer for source inspection.
 *
 * On load, fetches /api/options to sync defaults from the server config.
 */

(function () {
  "use strict";

  // --- DOM references ---
  const queryForm      = document.getElementById("query-form");
  const queryInput     = document.getElementById("query");
  const sendBtn        = document.getElementById("send-btn");
  const resultDiv      = document.getElementById("result");
  const debugMain      = document.getElementById("debug-main");
  const pdfPanel       = document.getElementById("pdf-panel");
  const pdfFrame       = document.getElementById("pdf-frame");
  const pdfTitle       = document.getElementById("pdf-panel-title");
  const pdfClose       = document.getElementById("pdf-panel-close");

  const elRetriever  = document.getElementById("retriever-type");
  const elKbBoxes    = document.getElementById("kb-checkboxes");
  const elStrategy   = document.getElementById("strategy");
  const elK          = document.getElementById("k-value");
  const elThreshold  = document.getElementById("threshold");

  const elOllamaDot  = document.getElementById("ollama-dot");
  const elOllamaStatus = document.getElementById("ollama-status");
  const elModelRow   = document.getElementById("model-row");
  const elModelLabel = document.getElementById("model-label");
  const elCacheRow   = document.getElementById("cache-row");
  const elCacheLabel = document.getElementById("cache-label");

  // ---------------------------------------------------------------------------
  // PDF panel
  // ---------------------------------------------------------------------------

  function openPdf(page, label) {
    const url = "/api/pdf/stpo#page=" + (page || 1);
    if (pdfFrame.src !== location.origin + url) {
      pdfFrame.src = url;
    }
    pdfTitle.textContent = "StPO \u2014 Page " + page + (label ? " \u00b7 " + label : "");
    pdfPanel.classList.add("open");
    debugMain.classList.add("pdf-open");
  }

  function closePdf() {
    pdfPanel.classList.remove("open");
    debugMain.classList.remove("pdf-open");
  }

  pdfClose.addEventListener("click", closePdf);

  // ---------------------------------------------------------------------------
  // API helpers
  // ---------------------------------------------------------------------------

  async function fetchOptions() {
    try {
      const r = await fetch("/api/options");
      return r.ok ? r.json() : null;
    } catch (_) { return null; }
  }

  async function fetchHealth() {
    try {
      const r = await fetch("/api/health");
      return r.ok ? r.json() : null;
    } catch (_) { return null; }
  }

  async function runQuery(query, cfg) {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...cfg }),
    });
    if (!resp.ok) {
      const d = await resp.json().catch(() => null);
      throw new Error(d && d.detail ? d.detail : "Request failed (" + resp.status + ")");
    }
    return resp.json();
  }

  // ---------------------------------------------------------------------------
  // Server status
  // ---------------------------------------------------------------------------

  async function refreshStatus() {
    const h = await fetchHealth();
    if (!h) {
      elOllamaDot.className = "status-dot unavailable";
      elOllamaStatus.textContent = "Ollama unreachable";
      return;
    }
    const connected = h.ollama === "connected";
    elOllamaDot.className = "status-dot " + (connected ? "connected" : "unavailable");
    elOllamaStatus.textContent = "Ollama " + h.ollama;
    elModelRow.style.display = "flex";
    elModelLabel.textContent = "Model: " + h.model;
    elCacheRow.style.display = "flex";
    elCacheLabel.textContent = "Cached retrievers: " + h.cached_retrievers;
  }

  // ---------------------------------------------------------------------------
  // Sync defaults from server config on load
  // ---------------------------------------------------------------------------

  async function syncDefaults() {
    const opts = await fetchOptions();
    if (!opts) return;

    const defs = opts.defaults || {};

    if (defs.retriever_type) elRetriever.value = defs.retriever_type;
    if (defs.strategy)       elStrategy.value  = defs.strategy;
    if (defs.k)              elK.value         = defs.k;

    if (defs.knowledge_bases && Array.isArray(defs.knowledge_bases)) {
      elKbBoxes.querySelectorAll("input[type=checkbox]").forEach(function (cb) {
        cb.checked = defs.knowledge_bases.includes(cb.value);
      });
    }

    // Add any KBs returned by the server that are not already listed
    if (opts.knowledge_bases) {
      const existing = new Set(
        Array.from(elKbBoxes.querySelectorAll("input")).map(function (i) { return i.value; })
      );
      opts.knowledge_bases.forEach(function (kb) {
        if (!existing.has(kb)) {
          const label = document.createElement("label");
          const cb    = document.createElement("input");
          cb.type  = "checkbox";
          cb.value = kb;
          cb.checked = (defs.knowledge_bases || []).includes(kb);
          label.appendChild(cb);
          label.appendChild(document.createTextNode(" " + kb));
          elKbBoxes.appendChild(label);
        }
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Build config from current UI state
  // ---------------------------------------------------------------------------

  function getConfig() {
    const kbs = Array.from(
      elKbBoxes.querySelectorAll("input:checked")
    ).map(function (cb) { return cb.value; });

    const cfg = {
      retriever_type:  elRetriever.value,
      knowledge_bases: kbs,
      strategy:        elStrategy.value,
      k:               parseInt(elK.value, 10) || 5,
    };

    const t = elThreshold.value.trim();
    if (t !== "") cfg.threshold = parseFloat(t);

    return cfg;
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function confClass(score) {
    if (score >= 0.60) return "conf-high";
    if (score >= 0.30) return "conf-med";
    return "conf-low";
  }

  function renderResult(data) {
    let html = "";

    // Config summary tags
    html += '<div class="config-tags">';
    html += '<span class="config-tag">' + esc(data.config.retriever_type) + '</span>';
    html += '<span class="config-tag">' + esc(data.config.strategy) + '</span>';
    html += '<span class="config-tag">k=' + data.config.k + '</span>';
    html += '<span class="config-tag">threshold=' + data.config.threshold + '</span>';
    html += '<span class="config-tag">' + esc(data.config.normalization_strategy) + '</span>';
    html += '<span class="config-tag">' + esc(data.config.model) + '</span>';
    (data.config.knowledge_bases || []).forEach(function (kb) {
      html += '<span class="config-tag">' + esc(kb) + '</span>';
    });
    html += '</div>';

    // Answer card
    html += '<div class="result-card">';
    html += '<div class="result-card-title">Answer</div>';

    if (data.abstained) {
      html +=
        '<div class="abstention-box">' +
          '<div class="abstention-title">Abstained (Level ' + (data.abstention_level || "?") + ')</div>' +
          '<div class="abstention-reason">' + esc(data.abstention_reason) + '</div>' +
        '</div>';
    } else {
      html += '<div class="answer-text">' + esc(data.answer) + '</div>';
    }

    const cls = confClass(data.confidence);
    html +=
      '<div style="margin-top:12px;display:flex;align-items:center;gap:10px;">' +
        '<span class="conf-badge ' + cls + '">' +
          (data.confidence * 100).toFixed(1) + '% confidence' +
        '</span>' +
        '<span style="font-size:12px;color:var(--c-text-faint);font-family:var(--font-mono);">' +
          '(' + data.confidence.toFixed(4) + ')' +
        '</span>' +
      '</div>';

    html += '</div>'; // /answer card

    // Sources table
    if (data.sources && data.sources.length > 0) {
      html += '<div class="result-card">';
      html += '<div class="result-card-title">Retrieved Chunks (' + data.sources.length + ')</div>';
      html += renderChunkTable(data.sources);
      html += '</div>';
    }

    resultDiv.innerHTML = html;

    // Attach PDF open listeners
    resultDiv.querySelectorAll("[data-open-pdf]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        openPdf(parseInt(btn.dataset.page, 10), btn.dataset.label);
      });
    });
  }

  function renderChunkTable(sources) {
    let html =
      '<table class="chunks-table">' +
      '<thead><tr>' +
        '<th class="col-num">#</th>' +
        '<th class="col-id">Chunk ID</th>' +
        '<th class="col-score">Score</th>' +
        '<th class="col-page">Page</th>' +
        '<th class="col-text">Text Preview</th>' +
      '</tr></thead><tbody>';

    sources.forEach(function (s, i) {
      const meta  = s.metadata || {};
      const page  = meta.start_page;
      const title = meta.section_title || "";

      let pageCell = '<td class="col-page">';
      if (page != null) {
        pageCell +=
          '<button class="source-page-btn" data-open-pdf data-page="' + page +
          '" data-label="' + esc(s.chunk_id) + '">' +
            'p.' + page +
          '</button>';
        if (meta.end_page && meta.end_page !== page) {
          pageCell += '<span style="font-size:11px;color:var(--c-text-faint);">&ndash;' + meta.end_page + '</span>';
        }
      } else {
        pageCell += '<span style="color:var(--c-text-faint);">—</span>';
      }
      pageCell += '</td>';

      html +=
        '<tr>' +
          '<td class="col-num">' + (i + 1) + '</td>' +
          '<td class="col-id">' +
            esc(s.chunk_id) +
            (title ? '<br><span style="font-size:10.5px;font-style:italic;color:var(--c-text-faint);font-family:var(--font);">' + esc(title) + '</span>' : '') +
          '</td>' +
          '<td class="col-score">' + s.score.toFixed(4) + '</td>' +
          pageCell +
          '<td class="col-text">' + esc(s.text.substring(0, 240)) + '</td>' +
        '</tr>';
    });

    html += '</tbody></table>';
    return html;
  }

  function renderLoading() {
    resultDiv.innerHTML =
      '<div class="loading-card"><span class="spinner"></span><span>Processing query&hellip;</span></div>';
  }

  function renderError(msg) {
    resultDiv.innerHTML =
      '<div class="result-card" style="border-color:var(--c-error);">' +
        '<div class="result-card-title" style="color:var(--c-error);">Error</div>' +
        '<div style="font-size:14px;">' + esc(msg) + '</div>' +
      '</div>';
  }

  function setLoading(loading) {
    queryInput.disabled = loading;
    sendBtn.disabled    = loading;
    if (!loading) queryInput.focus();
  }

  // ---------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------

  queryForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    setLoading(true);
    renderLoading();

    try {
      const cfg  = getConfig();
      const data = await runQuery(query, cfg);
      renderResult(data);
    } catch (err) {
      renderError(err.message);
    }

    setLoading(false);
    refreshStatus();
  });

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  syncDefaults();
  refreshStatus();

})();
