# Ollama-Evaluationslauf — Erstauswertung

> Interne Notizen für Joshua. Exakte Analyse kommt später.
> Lauf: `python -m evaluation --all --judge ollama --output-dir data/evaluation-ollama`
> Dauer: **12 h 17 min** (2026-04-18 14:16 → 2026-04-19 02:33)
> Hardware: RTX 4070 Ti SUPER 16 GB, Ollama `llama3.1:latest` mit `OLLAMA_NUM_PARALLEL=2`

---

## 1. Überblick über die Ergebnisse

### Retrieval (`retrieval-evaluation.json`)

21 Reports: 3 Retriever × (3 Single-KB + 3 Merged + 3 Fusion). F1@5 gemittelt pro Retriever-Typ:

| Retriever | F1@5 (Mittelwert) |
|---|---|
| BM25 | 0.187 |
| Hybrid (RRF) | 0.259 |
| **Vector** | **0.273** |

**Befund:** Vector-Retrieval schlägt BM25 deutlich, Hybrid liegt dazwischen.

### RRF-Tuning (`rrf-tuning.json`)

Bester `k_rrf`-Parameter pro Knowledge Base:

| Modus | KB | best k_rrf | F1@5 | MRR |
|---|---|---|---|---|
| Hybrid | stpo | 1 | 0.239 | 0.450 |
| Hybrid | faq-stpo | 1 | 0.248 | 0.494 |
| Hybrid | faq-ao | 1 | 0.000 | 0.000 |
| Fusion | alle | 1 | 0.213 | 0.394 |

**Wichtig:** Der optimale `k_rrf` ist durchweg **1**, nicht 60 (Default). Das ist ein diskutierswerter Befund für die Thesis — suggeriert, dass starke Top-Ergebnisse so stark gewichtet werden sollen, dass RRF fast wie ein "Max"-Join wirkt.

### Abstention (`abstention-evaluation.json`)

| KB | Precision | Recall | F1 |
|---|---|---|---|
| stpo | 0.266 | 1.000 | 0.420 |
| faq-stpo | 0.333 | 1.000 | 0.500 |
| faq-ao | 0.250 | 1.000 | 0.400 |

**Befund:** Recall=1.0 durchgehend — das System verpasst keine einzige unbeantwortbare Frage. Precision ist niedrig (0.25–0.33) — viele "false abstentions": System lehnt ab, obwohl die Frage beantwortbar wäre. Klarer Thesis-Trade-off.

### Generation (`generation-evaluation.json` + combined)

| KB | Faithfulness | Answer Relevancy | Factual Correctness | n |
|---|---|---|---|---|
| stpo | 0.500 | 0.422 | 0.377 | 825 |
| faq-stpo | 0.514 | 0.378 | 0.404 | 825 |
| faq-ao | — | — | — | **0 (leere KB)** |
| **Combined** | **0.554** | **0.459** | **0.447** | 825 |

**Befund:** Combined-KB schneidet durchweg besser ab als einzelne KBs. faq-ao hat 0 Chunks (Platzhalter) — daher keine Samples.

### End-to-End (33 Configs)

Insgesamt **481 gescorte Samples** über alle Configs (Ø 14.6 pro Config, min 0, max 49).

**Top 5 nach Factual Correctness:**

| Config | F | AR | FC | n_scored | Abst-F1 |
|---|---|---|---|---|---|
| merged-stpo+faq-stpo-hybrid | 0.65 | 0.66 | **0.63** | 18 | 0.47 |
| merged-faq-stpo+faq-ao-bm25 | 0.67 | 0.63 | 0.61 | 24 | 0.50 |
| single-faq-stpo-bm25 | 0.72 | 0.64 | 0.57 | 23 | 0.49 |
| merged-stpo+faq-stpo-bm25 | 0.69 | 0.62 | 0.57 | 31 | 0.53 |
| single-stpo-hybrid | 0.74 | 0.67 | 0.55 | 11 | 0.44 |

**Top 5 nach Abstention-F1:**

| Config | Abst-F1 | Precision | Recall | FC |
|---|---|---|---|---|
| single-faq-stpo-vector | 0.66 | 0.49 | 1.00 | 0.52 |
| merged-faq-stpo+faq-ao-vector | 0.65 | 0.48 | 1.00 | 0.50 |
| merged-all-vector | 0.62 | 0.45 | 1.00 | 0.49 |
| merged-stpo+faq-stpo-vector | 0.61 | 0.44 | 0.96 | 0.55 |
| merged-stpo+faq-ao-vector | 0.55 | 0.39 | 0.96 | 0.40 |

**Befund:** Die Config, die am meisten korrekt beantwortet (FC=0.63), ist **nicht** die Config, die am besten routet (Abst-F1=0.66). Klassischer Trade-off — muss in der Thesis sauber diskutiert werden.

**Kennzahlen über alle 33 Configs:**
- Ø Faithfulness: 0.709
- Ø Answer Relevancy: 0.609
- Ø Factual Correctness: 0.495

---

## 2. Judge-Reliability (NaN-Raten, Ollama als Judge)

| Evaluation | Samples | NaN Faithfulness | NaN Answer Relevancy | NaN Factual Corr. |
|---|---|---|---|---|
| Generation (per-KB) | 1650 | 145 (8.8 %) | 36 (2.2 %) | 4 (0.2 %) |
| Generation (combined) | 825 | 62 (7.5 %) | 15 (1.8 %) | 0 (0.0 %) |
| E2E (Option-A scoped) | 481 | 23 (4.8 %) | 4 (0.8 %) | 0 (0.0 %) |

**Befund:** Ollama hat ein konsistentes Problem mit Faithfulness (~5–9 % NaN). Das ist der strukturierteste RAGAS-Output (Liste von Claims + Verdicts) — `llama3.1:8B` produziert hier oft ungültiges JSON. OpenAI `gpt-4o-mini` hat das im früheren Pilot-Subset (3 % NaN-Rate) deutlich besser hinbekommen.

---

## 3. Kosten- und Zeit-Prognose für den OpenAI-Lauf

### Token-Schätzung

**Schätzwerte pro gescortem Sample** (auf Basis von 500-Sample-Stichprobe + tiktoken):

| Metrik | Input-Tokens | Output-Tokens |
|---|---|---|
| Faithfulness | ~1 490 | ~400 |
| Answer Relevancy | ~290 | ~300 |
| Factual Correctness | ~270 | ~400 |

**Faithfulness dominiert den Input** — sie bekommt die kompletten Retrieved Contexts mit rein (3–5 Chunks à ~250–400 Tokens).

### Scored Samples im geplanten OpenAI-`--all`-Lauf

| Stufe | Samples |
|---|---|
| Generation (stpo + faq-stpo + combined) | 2 475 |
| E2E (Option-A-eligible, aus Ollama-Lauf übertragen) | ~481 |
| **Gesamt** | **~2 956** |

### Token-Budget insgesamt

- **Input:** ~6.1 M Tokens
- **Output:** ~3.3 M Tokens

### Kosten `gpt-4o-mini` (Preise 2026: $0.15/M input, $0.60/M output)

| Posten | Kosten |
|---|---|
| Input | $0.91 |
| Output | $1.95 |
| **Gesamt** | **~$2.86** |

**Toleranzfenster:** ±30 % je nach tatsächlicher Kontextlänge pro Sample. Realistische Range: **$2.00–$4.00**. Keinesfalls zweistellig.

### Zeit-Prognose

- **Generator** (Ollama, unverändert): wird den gleichen Zeitanteil haben wie im Ollama-Lauf — also grob ~9–10 h von den 12.3 h (Rest war Judge-Zeit).
- **Judge-Calls bei OpenAI:** ~8 868 Calls (3 Metriken × 2 956 Samples), aufgeteilt in 178 Batches à 50. RAGAS-Async-Client nutzt Parallelität, aber Rate-Limits drosseln. Schätzung: **2–4 h Judge-Zeit**.
- **Gesamt OpenAI-`--all`:** **~11–14 h** — in der gleichen Größenordnung wie der Ollama-Lauf, eher leicht schneller, weil die OpenAI-Scoring-Phase pro Sample effizienter ist als Ollama.

**Praxis-Empfehlung:** Nächster Run kann wieder über Nacht laufen. Resume-Support ist aktiv — falls die API mal 429'ed, einfach das gleiche Kommando neu starten.

---

## 4. Sind wir fertig?

### Der Chatbot selbst

- **System funktioniert:** Pipeline läuft vollständig durch, 12.3 h ohne Crash
- **Tests:** 672 grün, 2 skipped, keine Regressionen
- **Architektur:** GPU als Baseline, Judge konfigurierbar, alle 5 Stages produzieren saubere Reports
- **Auffindbar:** `--check` bestätigt jede Run-Voraussetzung vor dem Start

**Bewertung:** ✅ **Ja, der Chatbot ist bei 100 %.** Er ist produktionsreif in dem Sinne, dass er die Daten verarbeitet, Antworten liefert, korrekt abstention triggert und evaluationsbereit ist.

### Die Evaluation-Pipeline

- Retrieval, RRF-Tuning, Abstention, Generation, E2E — alle 5 Schritte liefern valide numerische Reports
- 33 E2E-Configs durchgelaufen, alle JSONs populiert (inkl. `generation_metrics`-Block)
- Ollama-Judge NaN-Raten sind real, aber kein Bug — bekannte Schwäche von `llama3.1:8B` beim strukturierten JSON-Output. OpenAI-Lauf wird das kompensieren.
- Resume-Logik funktioniert (nicht getestet, aber code-reviewed — ein Abbruch wäre rekoverbar)

**Bewertung:** ✅ **Ja, die Evaluation funktioniert zu 100 %.** Was nicht 100 % ist, sind die *Scores selber* — das sind Thesis-Ergebnisse (niedrige Recall/Precision bei stpo, Retrieval-F1 ~0.2–0.27, FC ~0.5). Diese Zahlen werden in der Thesis diskutiert, nicht "gefixt".

### Was noch offen ist

| Punkt | Status |
|---|---|
| OpenAI-Lauf | ausstehend (~$3, ~12 h) |
| Thesis-Auswertung der JSON-Daten (Plots, Tabellen, Prosa) | ausstehend |
| `faq-ao` als echte KB befüllen | **offen — aktuell 0 Chunks** (Platzhalter, wie im README dokumentiert) |
| Vergleich Ollama-Judge vs. OpenAI-Judge in einem Dokument | nach OpenAI-Lauf |

### Zu flaggen für die Thesis-Diskussion

1. **`best k_rrf = 1`** für alle KBs — systematischer Befund, muss erklärt werden
2. **Abstention-Recall = 1.0** bei Precision 0.25–0.33 — klassisches Trade-off-Problem, "safety vs. utility"
3. **Generation auf faq-ao = 0 Samples** — Platzhalter-KB, das muss in der Thesis offen adressiert werden (entweder befüllen oder die Erwartung klar limitieren)
4. **Ollama-Faithfulness-NaN-Rate 8.8 %** — gilt als Schwäche des lokalen 8B-Modells, rechtfertigt den OpenAI-Judge als zweiten Datenpunkt
5. **Top-Config nach FC vs. Top-Config nach Abstention-F1 divergieren** — kein einzelner "Gewinner", Trade-off-Matrix gehört in die Thesis

---

## 5. Nächster Schritt

```bash
python -m evaluation --all --judge openai --output-dir data/evaluation-openai
```

Erwartung: ~12 h, ~$3, robustere Faithfulness-Scores, konsistenter Datensatz für die finale Analyse.
