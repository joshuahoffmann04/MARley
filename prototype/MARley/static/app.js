const traceStorageKey = "marley_debug_traces_v1";

const state = {
  options: null,
  selectedDocumentId: null,
  defaultRetrievalMode: "hybrid",
};

const elements = {
  readyPill: document.getElementById("ready-pill"),
  documentSelect: document.getElementById("document-select"),
  sourceList: document.getElementById("source-list"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  resetBtn: document.getElementById("reset-chat"),
};

function appendMessage(role, text, metaText = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `msg ${role}`;
  wrapper.textContent = text;

  if (metaText) {
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = metaText;
    wrapper.appendChild(meta);
  }

  elements.chatLog.appendChild(wrapper);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function readSelectedRetrievalMode() {
  const checked = document.querySelector("input[name='retrieval_mode']:checked");
  return checked ? checked.value : "hybrid";
}

function setRetrievalMode(mode) {
  const target = document.querySelector(`input[name='retrieval_mode'][value='${mode}']`);
  if (target) {
    target.checked = true;
  }
}

function getTraceHistory() {
  const raw = localStorage.getItem(traceStorageKey);
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}

function persistTrace(entry) {
  const history = getTraceHistory();
  history.unshift(entry);
  const capped = history.slice(0, 50);
  localStorage.setItem(traceStorageKey, JSON.stringify(capped));
}

function formatIsoDate(dateToken) {
  const parsed = new Date(dateToken);
  return Number.isNaN(parsed.getTime()) ? dateToken : parsed.toLocaleString("de-DE");
}

function renderSourcesForDocument(documentId) {
  const doc = state.options.documents.find((item) => item.document_id === documentId);
  elements.sourceList.innerHTML = "";
  if (!doc) {
    return;
  }

  doc.sources.forEach((source) => {
    const row = document.createElement("label");
    row.className = "source-item";

    const left = document.createElement("span");
    left.innerHTML = `<input type="checkbox" data-source-type="${source.source_type}" ${
      source.available ? "checked" : "disabled"
    } /> ${source.source_type}`;

    const right = document.createElement("small");
    right.textContent = source.available && source.last_modified ? formatIsoDate(source.last_modified) : "nicht vorhanden";

    row.appendChild(left);
    row.appendChild(right);
    elements.sourceList.appendChild(row);
  });
}

function selectedSourceTypes() {
  return Array.from(elements.sourceList.querySelectorAll("input[type='checkbox']:checked")).map((item) =>
    item.getAttribute("data-source-type")
  );
}

async function loadReadyStatus() {
  try {
    const response = await fetch("/api/ready");
    const payload = await response.json();
    const ready = payload.status === "ready";
    elements.readyPill.className = `pill ${ready ? "pill-ready" : "pill-degraded"}`;
    elements.readyPill.textContent = ready ? "Status: ready" : "Status: degraded";
  } catch (_error) {
    elements.readyPill.className = "pill pill-degraded";
    elements.readyPill.textContent = "Status: offline";
  }
}

async function loadOptions() {
  const response = await fetch("/api/options");
  if (!response.ok) {
    throw new Error(`Options request failed (${response.status})`);
  }
  const payload = await response.json();
  state.options = payload;
  state.defaultRetrievalMode = payload.default_retrieval_mode || "hybrid";

  elements.documentSelect.innerHTML = "";
  payload.documents.forEach((documentOption) => {
    const option = document.createElement("option");
    option.value = documentOption.document_id;
    option.textContent = documentOption.document_id;
    elements.documentSelect.appendChild(option);
  });

  state.selectedDocumentId = payload.default_document_id || payload.documents[0]?.document_id || null;
  if (state.selectedDocumentId) {
    elements.documentSelect.value = state.selectedDocumentId;
    renderSourcesForDocument(state.selectedDocumentId);
  }
  setRetrievalMode(state.defaultRetrievalMode);
}

async function sendChat(query) {
  const documentId = elements.documentSelect.value;
  const retrievalMode = readSelectedRetrievalMode();
  const sourceTypes = selectedSourceTypes();
  if (!sourceTypes.length) {
    throw new Error("Bitte mindestens eine verfügbare Quelle auswählen.");
  }

  const payload = {
    query,
    document_id: documentId,
    retrieval_mode: retrievalMode,
    source_types: sourceTypes,
  };
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Chat request failed (${response.status})`);
  }
  return body;
}

function initInteractions() {
  elements.documentSelect.addEventListener("change", (event) => {
    state.selectedDocumentId = event.target.value;
    renderSourcesForDocument(state.selectedDocumentId);
  });

  elements.resetBtn.addEventListener("click", () => {
    elements.chatLog.innerHTML = "";
    appendMessage(
      "assistant",
      "Chat wurde geleert. Stelle eine neue Frage.",
      "Single-Turn aktiv"
    );
  });

  elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = elements.chatInput.value.trim();
    if (!query) {
      return;
    }

    appendMessage("user", query);
    elements.chatInput.value = "";
    elements.sendBtn.disabled = true;

    try {
      const response = await sendChat(query);
      const meta = `${response.document_id} | ${response.retrieval_mode} | ${
        response.abstained ? "abstained" : "answered"
      }`;
      appendMessage("assistant", response.answer, meta);

      persistTrace({
        captured_at: new Date().toISOString(),
        query,
        response,
      });
    } catch (error) {
      appendMessage("assistant", `Fehler: ${error.message}`, "request_error");
    } finally {
      elements.sendBtn.disabled = false;
    }
  });
}

async function bootstrap() {
  appendMessage(
    "assistant",
    "Willkommen bei MARley. Wähle eine Studienordnung, Quellen und Retrieval-Modus, dann stelle deine Frage.",
    "Single-Turn Chat"
  );
  await loadReadyStatus();
  await loadOptions();
  initInteractions();
}

bootstrap().catch((error) => {
  appendMessage("assistant", `Initialisierung fehlgeschlagen: ${error.message}`, "fatal");
});
