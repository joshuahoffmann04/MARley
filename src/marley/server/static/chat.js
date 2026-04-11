/**
 * MARley Chat UI — Production interface.
 *
 * Design principle: each Q&A is rendered as an independent "query card"
 * (not a messenger thread) to make the stateless nature visually obvious.
 * The PDF slide-in panel opens when a user clicks a source page reference.
 *
 * Security note: All user-generated and API-returned text is escaped via
 * the esc() helper (textContent-based) before insertion into innerHTML.
 * No raw user input is ever inserted into the DOM unescaped.
 */

(function () {
  "use strict";

  var queryLog     = document.getElementById("query-log-inner");
  var welcomeState = document.getElementById("welcome-state");
  var chatForm     = document.getElementById("chat-form");
  var queryInput   = document.getElementById("query");
  var sendBtn      = document.getElementById("send-btn");
  var chatMain     = document.getElementById("chat-main");
  var pdfPanel     = document.getElementById("pdf-panel");
  var pdfFrame     = document.getElementById("pdf-frame");
  var pdfTitle     = document.getElementById("pdf-panel-title");
  var pdfClose     = document.getElementById("pdf-panel-close");

  var queryCount = 0;

  // ---------------------------------------------------------------------------
  // PDF panel
  // ---------------------------------------------------------------------------

  function openPdf(page) {
    // Always force-reload by clearing src first, then setting the new URL.
    // This fixes the bug where re-opening with a different page still showed
    // the old page because the browser cached the iframe src.
    var url = "/api/pdf/stpo#page=" + (page || 1);
    pdfFrame.src = "";
    // Use a microtask to ensure the browser registers the src change
    setTimeout(function () {
      pdfFrame.src = url;
    }, 0);
    pdfTitle.textContent = "StPO \u2014 Page " + page;
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
    var resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query }),
    });
    if (!resp.ok) {
      var detail = await resp.json().catch(function () { return null; });
      throw new Error(detail && detail.detail ? detail.detail : "Request failed (" + resp.status + ")");
    }
    return resp.json();
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Escape text for safe insertion into innerHTML (XSS prevention). */
  function esc(text) {
    var d = document.createElement("div");
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

  /** Determine the human-readable source name from chunk metadata. */
  function sourceName(source) {
    var meta = source.metadata || {};
    if (meta.faq_source) return meta.faq_source.toUpperCase();
    if (meta.source_file && meta.source_file.indexOf("msc-computer-science") !== -1) return "StPO";
    // Fallback: derive from chunk_id prefix
    var id = source.chunk_id || "";
    if (id.startsWith("faq-ao"))   return "FAQ-AO";
    if (id.startsWith("faq-stpo")) return "FAQ-STPO";
    return "StPO";
  }

  /** Check if a source is an FAQ chunk. */
  function isFaq(source) {
    var meta = source.metadata || {};
    return !!meta.faq_source || (source.chunk_id || "").startsWith("faq-");
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  /**
   * Render a complete query card.
   *
   * All dynamic text (question, answer, source text, chunk IDs, scores)
   * is escaped via esc() before innerHTML insertion. Only static markup
   * strings and pre-escaped values are used in innerHTML assignments.
   */
  function renderQueryCard(question, data) {
    queryCount += 1;
    var n = queryCount;

    // --- header ---
    var header = document.createElement("div");
    header.className = "query-card-header";
    header.innerHTML =
      '<span class="query-number">#' + n + '</span>' +
      '<span class="query-independent-tag">Independent query &mdash; no context from other questions</span>';

    // --- body ---
    var body = document.createElement("div");
    body.className = "query-body";

    // Question row (plain text, no highlight)
    body.innerHTML +=
      '<div class="query-question-row">' +
        '<div class="row-label">Your question</div>' +
        '<div class="query-question-text">' + esc(question) + '</div>' +
      '</div>';

    // Answer row (no dot before MARley)
    var answerHtml = '<div class="row-label">MARley</div>';
    if (data.abstained) {
      var level  = data.abstention_level || "?";
      var reason = data.abstention_reason || "No sufficient information found in the study regulations.";
      answerHtml +=
        '<div class="abstention-box">' +
          '<div class="abstention-title">Unable to answer (Level ' + esc(String(level)) + ')</div>' +
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

    // --- assemble card ---
    var card = document.createElement("div");
    card.className = "query-card";
    card.appendChild(header);
    card.appendChild(body);

    // --- footer: confidence + sources (hidden on abstention) ---
    if (!data.abstained) {
      var footer = document.createElement("div");
      footer.className = "query-card-footer";

      var cls   = confClass(data.confidence);
      var label = confLabel(data.confidence);
      var pct   = (data.confidence * 100).toFixed(1);
      footer.innerHTML =
        '<span class="conf-badge ' + cls + '">' + esc(label) + ' &mdash; ' + esc(pct) + '%</span>';

      var hasSources = data.sources && data.sources.length > 0;
      if (hasSources) {
        var srcBtn = document.createElement("button");
        srcBtn.className = "sources-btn";
        srcBtn.textContent = "Sources (" + data.sources.length + ")";
        footer.appendChild(srcBtn);
      }

      card.appendChild(footer);

      // --- sources list ---
      if (hasSources) {
        var sourcesList = document.createElement("div");
        sourcesList.className = "sources-list";

        data.sources.forEach(function (s) {
          var item = document.createElement("div");
          item.className = "source-item";

          var meta = s.metadata || {};
          var page = meta.start_page;
          var name = sourceName(s);
          var scoreStr = (s.score * 100).toFixed(1) + "%";
          var sourceRef = meta.source_reference || meta.section_label || "";

          // Source header: source name + confidence + optional reference
          var headerHtml =
            '<div class="source-item-header">' +
              '<span class="source-name">' + esc(name) +
                (sourceRef ? ' &middot; ' + esc(sourceRef) : '') +
              '</span>' +
              '<span class="source-meta">' +
                '<span class="source-score">' + esc(scoreStr) + '</span>';

          // StPO: "Open PDF" button
          if (page != null) {
            headerHtml +=
              '<button class="source-page-btn" data-page="' + parseInt(page, 10) + '">' +
                'Open PDF \u2192 p.' + parseInt(page, 10) +
              '</button>';
          }

          headerHtml += '</span></div>';

          if (isFaq(s)) {
            // FAQ sources: show question + answer text
            headerHtml +=
              '<div class="source-text-preview">' + esc(s.text) + '</div>';
          }
          // StPO sources: no text preview, just the button

          item.innerHTML = headerHtml;
          sourcesList.appendChild(item);
        });

        // Attach PDF open listeners
        sourcesList.querySelectorAll(".source-page-btn").forEach(function (btn) {
          btn.addEventListener("click", function () {
            openPdf(parseInt(btn.dataset.page, 10));
          });
        });

        card.appendChild(sourcesList);

        // Sources toggle
        srcBtn.addEventListener("click", function () {
          var open = sourcesList.classList.toggle("open");
          srcBtn.textContent = open
            ? "Hide sources"
            : "Sources (" + data.sources.length + ")";
        });
      }
    }

    return card;
  }

  function renderLoadingCard() {
    var el = document.createElement("div");
    el.className = "loading-card";
    el.id = "loading-card";
    el.innerHTML =
      '<span class="spinner"></span>' +
      '<span>Searching knowledge base and generating answer&hellip;</span>';
    return el;
  }

  function renderErrorCard(message) {
    var card = document.createElement("div");
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
    var log = document.getElementById("query-log");
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
    var query = queryInput.value.trim();
    if (!query) return;

    // Hide welcome state on first query
    if (welcomeState) welcomeState.style.display = "none";

    queryInput.value = "";
    queryInput.style.height = "auto";
    setLoading(true);

    var loadingEl = renderLoadingCard();
    queryLog.appendChild(loadingEl);
    scrollBottom();

    try {
      var data = await sendChat(query);
      loadingEl.remove();
      var card = renderQueryCard(query, data);
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
