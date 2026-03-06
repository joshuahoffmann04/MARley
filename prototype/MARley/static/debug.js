const traceStorageKey = "marley_debug_traces_v1";

const elements = {
  traceSelect: document.getElementById("trace-select"),
  clearBtn: document.getElementById("clear-traces"),
  generatorJson: document.getElementById("generator-json"),
  retrievalJson: document.getElementById("retrieval-json"),
  tabs: Array.from(document.querySelectorAll(".tab")),
};

function getHistory() {
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

function setHistory(history) {
  localStorage.setItem(traceStorageKey, JSON.stringify(history));
}

function prettifyJson(payload) {
  return JSON.stringify(payload, null, 2);
}

function selectTraceAt(index) {
  const history = getHistory();
  const selected = history[index];
  if (!selected) {
    elements.generatorJson.textContent = "Keine Daten vorhanden.";
    elements.retrievalJson.textContent = "Keine Daten vorhanden.";
    return;
  }

  elements.generatorJson.textContent = prettifyJson(selected.response.generator_response || {});
  elements.retrievalJson.textContent = prettifyJson(selected.response.retrieval_response || {});
}

function renderTraceList() {
  const history = getHistory();
  elements.traceSelect.innerHTML = "";

  history.forEach((entry, index) => {
    const option = document.createElement("option");
    const stamp = new Date(entry.captured_at).toLocaleString("de-DE");
    const query = String(entry.query || "").replace(/\s+/g, " ").trim();
    option.value = String(index);
    option.textContent = `${stamp} | ${query.slice(0, 72)}`;
    elements.traceSelect.appendChild(option);
  });

  if (history.length) {
    elements.traceSelect.selectedIndex = 0;
    selectTraceAt(0);
  } else {
    selectTraceAt(-1);
  }
}

function initTabs() {
  elements.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      elements.tabs.forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");

      const target = tab.getAttribute("data-target");
      if (target === "generator-json") {
        elements.generatorJson.classList.remove("hidden");
        elements.retrievalJson.classList.add("hidden");
      } else {
        elements.generatorJson.classList.add("hidden");
        elements.retrievalJson.classList.remove("hidden");
      }
    });
  });
}

function initInteractions() {
  elements.traceSelect.addEventListener("change", () => {
    selectTraceAt(Number(elements.traceSelect.value));
  });

  elements.clearBtn.addEventListener("click", () => {
    setHistory([]);
    renderTraceList();
  });
}

function bootstrap() {
  initTabs();
  initInteractions();
  renderTraceList();
}

bootstrap();
