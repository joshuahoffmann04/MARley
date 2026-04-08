/**
 * MARley Chat UI — Production interface.
 *
 * Design principle: each Q&A is rendered as an independent "query card"
 * (not a messenger thread) to make the stateless nature visually obvious.
 * The PDF slide-in panel opens when a user clicks a source page reference.
 */

(function () {
  "use strict";

  const queryLog     = document.getElementById("query-log-inner");
  const welcomeState = document.getElementById("welcome-state");
  const chatForm     = document.getElementById("chat-form");
  const queryInput   = document.getElementById("query");
  const sendBtn      = document.getElementById("send-btn");
  const chatMain     = document.getElementById("chat-main");
  const pdfPanel     = document.getElementById("pdf-panel");
  const pdfFrame     = document.getElementById("pdf-frame");
  const pdfTitle     = document.getElementById("pdf-panel-title");
  const pdfClose     = document.getElementById("pdf-panel-close");

  let queryCount = 0;

  // ---------------------------------------------------------------------------
  // PDF panel
  // ---------------------------------------------------------------------------

  function openPdf(page, chunkId) {
    const url = "/api/pdf/stpo#page=" + (page || 1);
    if (pdfFrame.src !== url) {
      pdfFrame.src = url;
    }
    pdfTitle.textContent = "StPO \u2014 Page " + page + (chunkId ? " \u00b7 " + chunkId : "");
    pdfPanel.classList.add("open");
    chatMain.classList.add("pdf-open");
  }

  function closePdf() {
    pdfPanel.classList.remove("open");
    chatMain.classList.remove("pdf-open");
  }

  pdfClose.addEventListener("click", closePdf);

  // ---------------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------------

  async function sendChat(query) {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => null);
      throw new Error(detail && detail.detail ? detail.detail : "Request failed (" + resp.status + ")");
    }
    return resp.json();
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function esc(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  function confClass(score) {
    if (score >= 0.60) return "conf-high";
    if (score >= 0.30) return "conf-med";
    return "conf-low";
  }

  function confLabel(score) {
    if (score >= 0.60) return "High confidence";
    if (score >= 0.30) return "Medium confidence";
    return "Low confidence";
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  function renderQueryCard(question, data) {
    queryCount += 1;
    const n = queryCount;

    // --- header ---
    const header = document.createElement("div");
    header.className = "query-card-header";
    header.innerHTML =
      '<span class="query-number">#' + n + '</span>' +
      '<span class="query-independent-tag">Independent query &mdash; no context from other questions</span>';

    // --- body ---
    const body = document.createElement("div");
    body.className = "query-body";

    // Question row
    body.innerHTML +=
      '<div class="query-question-row">' +
        '<div class="row-label">Your question</div>' +
        '<div class="query-question-text">' + esc(question) + '</div>' +
      '</div>';

    // Answer row
    let answerHtml = '<div class="row-label"><span class="answer-label-dot" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--c-primary);margin-right:5px;"></span>MARley</div>';
    if (data.abstained) {
      const level  = data.abstention_level || "?";
      const reason = data.abstention_reason || "No sufficient information found in the study regulations.";
      answerHtml +=
        '<div class="abstention-box">' +
          '<div class="abstention-title">Unable to answer (Level ' + level + ')</div>' +
          '<div class="abstention-reason">' + esc(reason) + '</div>' +
          '<div class="abstention-hint">' +
            'For questions not covered by the study regulations, please contact the ' +
            'Academic Advisory Office (Studienberatung) directly.' +
          '</div>' +
        '</div>';
    } else {
      answerHtml += '<div class="answer-text">' + esc(data.answer) + '</div>';
    }
    body.innerHTML += answerHtml;

    // --- footer ---
    const footer = document.createElement("div");
    footer.className = "query-card-footer";

    const cls   = confClass(data.confidence);
    const label = confLabel(data.confidence);
    const pct   = (data.confidence * 100).toFixed(1);
    footer.innerHTML =
      '<span class="conf-badge ' + cls + '">' + label + ' &mdash; ' + pct + '%</span>';

    const hasSources = data.sources && data.sources.length > 0;
    if (hasSources) {
      const srcBtn = document.createElement("button");
      srcBtn.className = "sources-btn";
      srcBtn.textContent = "Sources (" + data.sources.length + ")";
      footer.appendChild(srcBtn);
    }

    // --- sources list ---
    const sourcesList = document.createElement("div");
    sourcesList.className = "sources-list";

    if (hasSources) {
      data.sources.forEach(function (s) {
        const item = document.createElement("div");
        item.className = "source-item";

        const meta   = s.metadata || {};
        const page   = meta.start_page;
        const title  = meta.section_title || "";

        let headerHtml =
          '<div class="source-item-header">' +
            '<span class="source-chunk-id">' + esc(s.chunk_id) + '</span>' +
            '<span class="source-meta">' +
              '<span class="source-score">' + s.score.toFixed(4) + '</span>';

        if (page != null) {
          headerHtml +=
            '<button class="source-page-btn" data-page="' + page + '" data-id="' + esc(s.chunk_id) + '">' +
              'Open PDF \u2192 p.' + page +
            '</button>';
        }

        headerHtml += '</span></div>';

        if (title) {
          headerHtml += '<div class="source-section">' + esc(title) + '</div>';
        }

        headerHtml +=
          '<div class="source-text-preview">' + esc(s.text.substring(0, 280)) + '</div>';

        item.innerHTML = headerHtml;
        sourcesList.appendChild(item);
      });

      // Attach PDF open listeners
      sourcesList.querySelectorAll(".source-page-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          openPdf(parseInt(btn.dataset.page, 10), btn.dataset.id);
        });
      });
    }

    // --- assemble card ---
    const card = document.createElement("div");
    card.className = "query-card";
    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(footer);
    card.appendChild(sourcesList);

    // Sources toggle
    if (hasSources) {
      const srcBtn = footer.querySelector(".sources-btn");
      srcBtn.addEventListener("click", function () {
        const open = sourcesList.classList.toggle("open");
        srcBtn.textContent = open
          ? "Hide sources"
          : "Sources (" + data.sources.length + ")";
      });
    }

    return card;
  }

  function renderLoadingCard() {
    const el = document.createElement("div");
    el.className = "loading-card";
    el.id = "loading-card";
    el.innerHTML =
      '<span class="spinner"></span>' +
      '<span>Searching knowledge base and generating answer&hellip;</span>';
    return el;
  }

  function renderErrorCard(message) {
    const card = document.createElement("div");
    card.className = "query-card";
    card.innerHTML =
      '<div class="query-card-header">' +
        '<span class="query-number" style="background:var(--c-error);">#' + (queryCount + 1) + '</span>' +
        '<span class="query-independent-tag" style="color:var(--c-error);">Error</span>' +
      '</div>' +
      '<div class="query-body">' +
        '<div class="answer-text" style="color:var(--c-error);">' + esc(message) + '</div>' +
      '</div>';
    return card;
  }

  function scrollBottom() {
    const log = document.getElementById("query-log");
    log.scrollTop = log.scrollHeight;
  }

  function setLoading(loading) {
    queryInput.disabled = loading;
    sendBtn.disabled = loading;
    if (!loading) queryInput.focus();
  }

  // ---------------------------------------------------------------------------
  // Auto-resize textarea
  // ---------------------------------------------------------------------------

  queryInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  // ---------------------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------------------

  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Hide welcome state on first query
    if (welcomeState) welcomeState.style.display = "none";

    queryInput.value = "";
    queryInput.style.height = "auto";
    setLoading(true);

    const loadingEl = renderLoadingCard();
    queryLog.appendChild(loadingEl);
    scrollBottom();

    try {
      const data = await sendChat(query);
      loadingEl.remove();
      const card = renderQueryCard(query, data);
      queryLog.appendChild(card);
    } catch (err) {
      loadingEl.remove();
      queryLog.appendChild(renderErrorCard("Error: " + err.message));
    }

    setLoading(false);
    scrollBottom();
  });

  // Enter to submit, Shift+Enter for newline
  queryInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

})();
