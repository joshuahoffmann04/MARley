/**
 * MARley Manual Evaluation — Frontend Logic
 *
 * Handles item loading, navigation, judgement submission,
 * filtering, and keyboard shortcuts.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
    source: "",
    items: [],       // [{item: {...}, judgement: {...}|null}, ...]
    currentIndex: 0,
    loading: false,
};

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const dom = {
    filterSource: document.getElementById("filter-source"),
    filterDistractors: document.getElementById("filter-distractors"),
    filterCategory: document.getElementById("filter-category"),
    filterStatus: document.getElementById("filter-status"),
    progressText: document.getElementById("progress-text"),
    progressFill: document.getElementById("progress-fill"),
    noItems: document.getElementById("no-items"),
    itemDisplay: document.getElementById("item-display"),
    itemIndex: document.getElementById("item-index"),
    itemId: document.getElementById("item-id"),
    itemBadge: document.getElementById("item-judgement-badge"),
    questionText: document.getElementById("question-text"),
    generatedAnswer: document.getElementById("generated-answer"),
    referenceAnswer: document.getElementById("reference-answer"),
    metaKb: document.getElementById("meta-kb"),
    metaDistractors: document.getElementById("meta-distractors"),
    metaCategory: document.getElementById("meta-category"),
    metaAbstention: document.getElementById("meta-abstention"),
    notesInput: document.getElementById("notes-input"),
    btnPrev: document.getElementById("btn-prev"),
    btnNext: document.getElementById("btn-next"),
    answerJudgements: document.getElementById("answer-judgements"),
    abstentionJudgements: document.getElementById("abstention-judgements"),
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    return resp.json();
}

async function postJSON(url, body) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    return resp.json();
}

// ---------------------------------------------------------------------------
// Source loading
// ---------------------------------------------------------------------------

async function loadSources() {
    const sources = await fetchJSON("api/sources");
    dom.filterSource.innerHTML = "";

    if (sources.length === 0) {
        dom.filterSource.innerHTML = '<option value="">No sources found</option>';
        return;
    }

    for (const s of sources) {
        const opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = s.name;
        dom.filterSource.appendChild(opt);
    }

    state.source = sources[0].name;
    await loadItems();
}

// ---------------------------------------------------------------------------
// Item loading
// ---------------------------------------------------------------------------

async function loadItems() {
    if (state.loading) return;
    state.loading = true;

    const source = dom.filterSource.value;
    state.source = source;

    if (!source) {
        state.items = [];
        renderCurrentItem();
        state.loading = false;
        return;
    }

    let url = `api/items?source=${encodeURIComponent(source)}`;

    const distractors = dom.filterDistractors.value;
    if (distractors !== "") url += `&filter_distractors=${distractors}`;

    const category = dom.filterCategory.value;
    if (category) url += `&filter_category=${encodeURIComponent(category)}`;

    const status = dom.filterStatus.value;
    if (status) url += `&filter_status=${encodeURIComponent(status)}`;

    try {
        state.items = await fetchJSON(url);
        state.currentIndex = 0;
        renderCurrentItem();
        await updateProgress();
    } catch (err) {
        console.error("Failed to load items:", err);
    }

    state.loading = false;
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

async function updateProgress() {
    if (!state.source) return;
    try {
        const progress = await fetchJSON(`api/progress?source=${encodeURIComponent(state.source)}`);
        const pct = progress.total > 0 ? (progress.judged / progress.total * 100) : 0;
        dom.progressText.textContent = `${progress.judged} / ${progress.total}`;
        dom.progressFill.style.width = `${pct}%`;
    } catch (err) {
        console.error("Failed to load progress:", err);
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderCurrentItem() {
    if (state.items.length === 0) {
        dom.noItems.classList.remove("hidden");
        dom.itemDisplay.classList.add("hidden");
        return;
    }

    dom.noItems.classList.add("hidden");
    dom.itemDisplay.classList.remove("hidden");

    const entry = state.items[state.currentIndex];
    const item = entry.item;
    const judgement = entry.judgement;

    // Header
    dom.itemIndex.textContent = `${state.currentIndex + 1} / ${state.items.length}`;
    dom.itemId.textContent = item.id;

    // Judgement badge
    if (judgement) {
        dom.itemBadge.classList.remove("hidden");
        dom.itemBadge.textContent = judgement.judgement.replace(/_/g, " ");
        dom.itemBadge.className = `${judgement.judgement}`;
        dom.itemBadge.id = "item-judgement-badge";
    } else {
        dom.itemBadge.classList.add("hidden");
    }

    // Question and answers
    dom.questionText.textContent = item.question || "(no question text)";
    dom.generatedAnswer.textContent = item.generated_answer;
    dom.referenceAnswer.textContent = item.reference_answer;

    // Metadata
    const meta = item.metadata || {};
    dom.metaKb.textContent = `KB: ${meta.knowledge_base || "—"}`;
    dom.metaDistractors.textContent = `Distractors: ${meta.num_distractors ?? "—"}`;
    dom.metaCategory.textContent = `Category: ${item.category || "—"}`;
    dom.metaAbstention.textContent = item.expected_abstention ? "Expected: Abstention" : "Expected: Answer";

    // Notes
    dom.notesInput.value = judgement ? (judgement.notes || "") : "";

    // Highlight appropriate judgement row
    if (item.expected_abstention) {
        dom.answerJudgements.classList.add("dimmed");
        dom.answerJudgements.classList.remove("suggested");
        dom.abstentionJudgements.classList.add("suggested");
        dom.abstentionJudgements.classList.remove("dimmed");
    } else {
        dom.answerJudgements.classList.add("suggested");
        dom.answerJudgements.classList.remove("dimmed");
        dom.abstentionJudgements.classList.add("dimmed");
        dom.abstentionJudgements.classList.remove("suggested");
    }

    // Highlight active judgement button
    document.querySelectorAll(".btn-judgement").forEach(btn => {
        btn.classList.remove("active");
        if (judgement && btn.dataset.judgement === judgement.judgement) {
            btn.classList.add("active");
        }
    });

    // Navigation buttons
    dom.btnPrev.disabled = state.currentIndex === 0;
    dom.btnNext.disabled = state.currentIndex >= state.items.length - 1;
}

// ---------------------------------------------------------------------------
// Judgement submission
// ---------------------------------------------------------------------------

async function submitJudgement(judgementValue) {
    if (state.items.length === 0) return;

    const entry = state.items[state.currentIndex];
    const itemId = entry.item.id;
    const notes = dom.notesInput.value.trim();

    try {
        await postJSON(`api/judgements?source=${encodeURIComponent(state.source)}`, {
            item_id: itemId,
            judgement: judgementValue,
            notes: notes,
        });

        // Update local state
        entry.judgement = {
            item_id: itemId,
            judgement: judgementValue,
            notes: notes,
            timestamp: new Date().toISOString(),
        };

        renderCurrentItem();
        await updateProgress();

        // Auto-advance to next unjudged item
        advanceToNextUnjudged();
    } catch (err) {
        console.error("Failed to save judgement:", err);
        alert("Failed to save judgement. See console for details.");
    }
}

function advanceToNextUnjudged() {
    // Find the next unjudged item after current index
    for (let i = state.currentIndex + 1; i < state.items.length; i++) {
        if (!state.items[i].judgement) {
            state.currentIndex = i;
            renderCurrentItem();
            return;
        }
    }
    // If no unjudged after current, try from beginning
    for (let i = 0; i < state.currentIndex; i++) {
        if (!state.items[i].judgement) {
            state.currentIndex = i;
            renderCurrentItem();
            return;
        }
    }
    // All judged — just move to next if possible
    if (state.currentIndex < state.items.length - 1) {
        state.currentIndex++;
        renderCurrentItem();
    }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function navigate(direction) {
    const newIndex = state.currentIndex + direction;
    if (newIndex >= 0 && newIndex < state.items.length) {
        state.currentIndex = newIndex;
        renderCurrentItem();
    }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

// Filter changes
dom.filterSource.addEventListener("change", loadItems);
dom.filterDistractors.addEventListener("change", loadItems);
dom.filterCategory.addEventListener("change", loadItems);
dom.filterStatus.addEventListener("change", loadItems);

// Navigation buttons
dom.btnPrev.addEventListener("click", () => navigate(-1));
dom.btnNext.addEventListener("click", () => navigate(1));

// Judgement buttons
document.querySelectorAll(".btn-judgement").forEach(btn => {
    btn.addEventListener("click", () => submitJudgement(btn.dataset.judgement));
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
    // Don't capture shortcuts when typing in the notes field
    if (e.target === dom.notesInput) return;

    const keyMap = {
        "1": "correct",
        "2": "partially_correct",
        "3": "incorrect",
        "4": "correct_abstention",
        "5": "incorrect_abstention",
        "6": "missing_abstention",
    };

    if (keyMap[e.key]) {
        e.preventDefault();
        submitJudgement(keyMap[e.key]);
    } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        navigate(-1);
    } else if (e.key === "ArrowRight") {
        e.preventDefault();
        navigate(1);
    }
});

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

loadSources();
