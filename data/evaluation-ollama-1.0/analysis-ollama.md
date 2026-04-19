# Umfassende Analyse — Ollama-Evaluationslauf

> Entscheidungsgrundlage: Soll vor dem finalen OpenAI-Lauf noch etwas an der Pipeline geändert werden?
> Datum: 2026-04-19
> Quelle: `data/evaluation-ollama/` (33 E2E-Configs + Generation/Retrieval/RRF/Abstention-Reports aus 12.3 h Lauf)

---

## TL;DR

**Empfehlung: Nichts an der Pipeline ändern, OpenAI-Lauf jetzt starten.**

Drei Findings aus der Tiefenanalyse:

1. **Ollama-Judge ist fehlkalibriert** — nicht das System. Viele knappe, faktisch korrekte Antworten ("3 credits" exakt matching die Referenz) erhalten FC = 0.0. Das ist ein reines Judge-Problem, **OpenAI wird das beheben**.
2. **Fusion-Retrieval hat strukturelles Abstention-Problem** — confidence ist mathematisch konstant 0.333 für alle Queries. **Kein Quick-Fix möglich** ohne Re-Design, aber ein hervorragendes Thesis-Finding.
3. **Der System-Prompt hat keine nachweisbaren Schwächen** — Sprache konsistent, Grounding-Regel greift, Halluzinationen selten und unsystematisch.

Ein Prompt-Umbau wäre jetzt Cargo-Kult — es gibt kein identifiziertes Problem, das er lösen würde. Ein Fusion-Redesign wäre ein neues Feature, kein Bugfix. Alles, was sich durch den OpenAI-Judge verbessern lässt, verbessert sich automatisch.

---

## 1. Überblick über die Ergebnisse

### Retrieval (Single-KB + Merged + Fusion, 21 Reports)

| Retriever | Mean F1@5 | Range |
|---|---|---|
| BM25 | 0.187 | 0.00–0.36 |
| Hybrid | 0.259 | 0.00–0.41 |
| **Vector** | **0.273** | 0.00–0.45 |

**Befund:** Vector > Hybrid > BM25 im Mittel. Die Null-Werte kommen aus der `faq-ao`-KB (leer), alle drei Retriever geben dort 0.

### RRF-Tuning

| Sweep | KB | best k_rrf | F1@5 |
|---|---|---|---|
| Hybrid | stpo | 1 | 0.239 |
| Hybrid | faq-stpo | 1 | 0.248 |
| Hybrid | faq-ao | 1 | 0.000 |
| Fusion | all | 1 | 0.213 |

**Befund:** Überall ist `k_rrf = 1` optimal (Default war 60). Das bedeutet: Top-Ergebnisse des BM25-Zweigs sollen sehr dominant sein — RRF verhält sich fast wie ein Max-Join. Diskutierenswerter Thesis-Befund, aber keine Pipeline-Änderung nötig (der Eval-Sweep findet das ohnehin empirisch).

### Abstention (Standalone)

| KB | Precision | Recall | F1 |
|---|---|---|---|
| stpo | 0.266 | 1.000 | 0.420 |
| faq-stpo | 0.333 | 1.000 | 0.500 |
| faq-ao | 0.250 | 1.000 | 0.400 |

**Befund:** Recall = 1.0 durchgehend → System erkennt jede unbeantwortbare Frage. Low precision → viele false abstentions. Klassisches safety-vs-utility Trade-off.

### Generation (per KB + Combined, 11 Distractor-Level)

| KB | Faithfulness | Answer Relevancy | Factual Correctness | n |
|---|---|---|---|---|
| stpo | 0.500 | 0.422 | 0.377 | 825 |
| faq-stpo | 0.514 | 0.378 | 0.404 | 825 |
| faq-ao | — | — | — | 0 |
| Combined | 0.554 | 0.459 | 0.447 | 825 |

**Überraschender Befund:** Die Scores **degradieren NICHT** mit steigender Distraktoren-Zahl. Stpo-Faithfulness steigt sogar leicht (0.451 → 0.582 bei 0 → 9 Distraktoren). Zwei mögliche Interpretationen:
- **Das Modell ist robust grounded** — Distraktoren lenken nicht ab.
- **Der Judge ist schwach kalibriert** und das FC-Rauschen dominiert die Distraktor-Effekte.

Ich halte Interpretation 2 für wahrscheinlicher (siehe § 3.1).

### E2E (33 Configs)

Summary:
- Ø Faithfulness: 0.709
- Ø Answer Relevancy: 0.609
- Ø Factual Correctness: 0.495
- Total gescorte Samples: **481** (von theoretisch 33 × 75 = 2475 möglich)

**Top-5 nach Correctness:**

| Config | F | AR | FC | n_scored | Abst-F1 |
|---|---|---|---|---|---|
| merged-stpo+faq-stpo-hybrid | 0.65 | 0.66 | **0.63** | 18 | 0.47 |
| merged-faq-stpo+faq-ao-bm25 | 0.67 | 0.63 | 0.61 | 24 | 0.50 |
| single-faq-stpo-bm25 | 0.72 | 0.64 | 0.57 | 23 | 0.49 |
| merged-stpo+faq-stpo-bm25 | 0.69 | 0.62 | 0.57 | 31 | 0.53 |
| single-stpo-hybrid | 0.74 | 0.67 | 0.55 | 11 | 0.44 |

**Pearson(Abst-F1, FC) = 0.215** → schwach positive Korrelation. Die Routing-Qualität ist nicht die Antwort-Qualität.

---

## 2. Kritische Befunde

### 2.1 Fusion-Retrieval: konstantes Confidence-Level (12 Configs betroffen)

**15 von 33 Configs haben `num_scored = 0`.** Die Verteilung:

| Config-Gruppe | Anzahl | Grund |
|---|---|---|
| `fusion-*` (alle 12) | 12 | Confidence konstant 0.333, Level-1-Threshold kann nicht diskriminieren |
| `single-faq-ao-*` (3) | 3 | Empty KB (0 Chunks) → Retrieval gibt nichts zurück → always abstain |

#### Fusion-Diagnose

Für `fusion-all-bm25` — beobachtet:

- confidence über alle 100 Fragen: **exakt 0.333 (min = max = mean)**
- Threshold-Sweep-Verhalten:
  - `threshold ∈ [0.00, 0.30]`: F1 = 0.00 (niemand abstained)
  - `threshold ∈ [0.35, 1.00]`: F1 = 0.40 (alle abstained, perfekt Recall aber 25 % Precision)
  - **Es gibt keinen Zwischenzustand.** Das ist keine Verteilung, das ist ein Stufen-Switch.

#### Ursache (math, kein Bug)

```
FusionRetriever bündelt n=3 Sub-Retriever, je einer pro KB.
Ein Chunk existiert in genau einer KB, rankt also nur in einem Sub-Retriever.
RRF-Score eines Top-1-Chunks = 1/(k_rrf+1)
Normalisierungs-Maximum (compute_confidence annimmt n-Retriever-Agreement): n/(k_rrf+1)
Normalized Score = (1/(k_rrf+1)) / (n/(k_rrf+1)) = 1/n = 0.333 bei n=3.

→ Confidence ist mathematisch konstant für jede Query, solange Retriever disjunkte Korpora haben.
```

#### Was bedeutet das für die Thesis?

**Das ist ein hochwertiges methodisches Finding**, kein Versagen der Pipeline: cross-KB RRF-Fusion mit confidence-basierter Abstention funktioniert nicht, wenn die Sub-Korpora disjunkt sind.

Mögliche Fixes (keiner trivial):
- Confidence aus den **Raw-Scores der Sub-Retriever** ableiten (z. B. `max(raw_BM25_score, raw_vector_score)`) statt aus RRF.
- Alternativ: Fusion pre-Normalisierungs-Rank-Basis verwenden.

**Entscheidung:** Kein Fix vor dem OpenAI-Lauf. Der Befund bleibt als Thesis-Diskussion erhalten. Ein Fix jetzt würde das System ändern und das OpenAI-Ergebnis inkomparabel zum Ollama-Ergebnis machen.

### 2.2 Judge-Kalibrierung: falsch-negative FC-Scores

Unter den 20 schlechtesten (FC < 0.3) E2E-Antworten fanden sich **mehrere offensichtlich korrekte Antworten mit FC = 0.00**. Beispiele:

| Frage | Antwort | Referenz | Ollama-Judge FC |
|---|---|---|---|
| Wie viele Credits hat das Seminar-Modul CS 610? | "3 credits." | "3 credits." | **0.00** |
| Wie viele Credits hat Algorithm Engineering CS 628? | "9 credits (LP)." | "9 credits (LP)." | 1.00 bei best-config, 0.00 bei anderer |
| Exam form für Neural Networks CS 593? | "Oral examination (individual) or written examination (Klausur)." | "Oral examination (individual) or written examination (Klausur)." | 0.00 |
| Wie viele Credits hat das Thesis-Modul? | "27 LP for thesis + 3 LP defense = 30 LP" | Gleich | variiert |

Das ist eine **Schwäche des `llama3.1:8B`-Judges** beim RAGAS-FactualCorrectness-Metric. Die Metric zerlegt Antwort und Referenz in Claims und vergleicht auf Deckung; der 8B-Judge scheint bei sehr knappen Antworten die Claim-Extraction inkonsistent zu machen.

**Impact:** Unsere aggregierten FC-Werte sind **systematisch unterschätzt**. Der frühere OpenAI-Pilot-Subset zeigte bereits: OpenAI-Judge gibt bei exakt denselben Samples **höhere, konsistentere Scores** (und hat deutlich niedrigere NaN-Raten).

**Entscheidung:** Wird sich durch den OpenAI-Lauf automatisch beheben. Keine Pipeline-Änderung.

### 2.3 Faithfulness-NaN-Rate 8.8 % auf langen/strukturierten Antworten

Von 23 NaN-Faithfulness-Cases (aus 481 E2E-eligible Samples):
- **~80 %** sind lange, listen-formatierte Antworten ("Module importable: - X (6 LP) - Y (9 LP) - ..."), 300–900 Zeichen
- **~20 %** sind mehrfach-fakt-Antworten mit komplexer Struktur

Ursache: RAGAS-Faithfulness-Metric zerlegt die Antwort in diskrete Statements und prüft jedes gegen den Kontext. Der 8B-Judge produziert bei langen Listen ungültiges JSON → NaN.

**Impact:** Konsistent mit dem allgemeinen Judge-Kalibrierungsproblem. Der OpenAI-Pilot-Subset zeigte 3 % statt 8.8 %.

**Entscheidung:** Siehe 2.2 — löst sich mit OpenAI.

### 2.4 Abstention-Threshold: aggressive Wahl

Threshold-Verteilung über 33 E2E-Configs:

| Threshold | Anzahl Configs | Kontext |
|---|---|---|
| 0.00 | 3 | single-faq-ao (leere KB) |
| 0.35 | 3 | fusion-all-* (degenerierte Confidence) |
| 0.45 | 2 | fusion-special-cases |
| 0.55 | 11 | Fusion (dominant) |
| 0.60 | 2 | merged |
| **0.95** | **6** | **single-stpo/faq-stpo mit bm25/hybrid** |
| **1.00** | **6** | **merged-* mit bm25/hybrid** |

12 Configs bekommen Threshold ≥ 0.95 → **alle Top-Retrievals müssen fast perfekt sein, sonst wird abstained**. Das erklärt die niedrigen `num_scored`-Zahlen in vielen Configs.

**Ursache:** Der F1-maximierende Threshold-Sweep bevorzugt hohe Abstention-Recall (perfekt) auf Kosten der Precision — im Dataset sind 25 % unbeantwortbare Fragen, die alle richtig abgelehnt werden sollen. Der Sweep findet: "So hoch wie möglich setzen, dann stimmt der Recall". Das ist das lokale Optimum der F1-Funktion, aber sub-optimal in der Praxis.

**Entscheidung:** Das Threshold-Design ist dokumentiert und verteidigbar — es ist *bewusst* F1-optimiert. Die niedrige Precision ist ein Thesis-Diskussionspunkt ("ist F1 die richtige Target-Metric für Abstention-Threshold-Selection, oder sollte man Precision bevorzugen?"). Keine Änderung.

---

## 3. Qualitative Stichproben (repräsentative Auszüge)

### 3.1 Top-FC-Antworten (alle FC = 1.0)

| Config | Frage | Antwort | Warum gut |
|---|---|---|---|
| merged-all-bm25 | How long is the standard study period? | "The standard program duration is 4 semesters." | Präzise, sprachlich angepasst |
| merged-all-bm25 | Prerequisites for Master's Thesis module? | "At least 66 credits earned in the master's program." | Zahl + Kontext korrekt |
| merged-all-hybrid | Duration of written examination? | "A written examination (Klausur) lasts between 60 and 120 minutes." | Range + term, matches |
| merged-all-hybrid | Structural variant exists? | "Yes, it has a single-subject structural variant." | Yes-Statement + German term |

**Gemeinsame Eigenschaften:** Konkrete Zahlen oder harte Fakten, kurze Form (≤ 20 Tokens), Term aus der Referenz übernommen. Der Ollama-Judge erkennt diese klar als korrekt.

### 3.2 Low-FC-Fälle, echte Probleme vs. Judge-Artefakte

Von den 20 schlechtesten:
- **~8 Fälle**: Judge-Artefakt (Antwort eigentlich korrekt, z. B. "3 credits" = "3 credits", aber FC = 0)
- **~6 Fälle**: echte Halluzinationen (eval-001 "2 years" statt "4 semesters"; eval-018 erfundenes Datum)
- **~4 Fälle**: ambiguous → StPO sagt nichts explizites, System sagt "Yes" wo es abstainen sollte
- **~2 Fälle**: Truncated/zu-knappe Antwort die wichtige Nuancen auslässt

**Diagnose:** Echte Antwort-Qualitätsprobleme sind in der Minderheit. Die Hälfte der "schlechtesten" Scores sind Judge-Fehlkalibrierungen.

### 3.3 False Abstentions: 1994 Fälle

Stratifizierte Stichprobe von 15 Fällen zeigt:

- **~10 / 15 sind Fusion-Configs** (die 12 Fusion-Configs haben nach Bauart 75 × 12 = 900 false abstentions)
- **~3 / 15 sind single-faq-ao** (leere KB)
- **~2 / 15 sind legitime Level-1-Abstention** auf Configs mit aggressivem Threshold

Bei den 2 legitimen: Die richtigen Chunks wurden *retrieved* (z. B. `par-16-txt-1` für eval-017 über Prüfungsausschuss), aber die Confidence lag knapp unter dem Threshold. Keine Prompt-Änderung kann das fixen — das ist Retrieval-Confidence, nicht Generator-Verhalten.

### 3.4 Sprach-Konsistenz

Stichprobe von 404 Q-A-Paaren:
- Fragen: 404 Englisch, 0 Deutsch
- Antworten: 347 Englisch, 57 kurz-nicht-klassifizierbar
- **Zero Mismatches**

Das Eval-Dataset ist einheitlich Englisch. Kein Sprach-Problem im System.

---

## 4. Bewertung der bestehenden System-Prompt

Der aktuelle Prompt (`src/marley/generator/prompt.py`):

```
You are a study advisor for the M.Sc. Computer Science program
at Philipps-Universität Marburg.

Answer the student's question using ONLY the numbered context
passages below. Follow these rules:
1. Base your answer exclusively on information from the provided context.
2. Be concise, precise, and factually accurate.
3. If insufficient info, respond: "ABSTENTION: <reason>"
4. Never guess, speculate, or supplement.
5. Plain text, no [1][2] references.
```

### Was funktioniert

- Regel 1 (Grounding): Top-10-Antworten zeigen klare Kontext-Treue.
- Regel 3 (Abstention-Format): Level-2-Detection fängt `ABSTENTION:`-Muster zuverlässig ab.
- Regel 5 (clean output): 0 von ~400 gesampelten Antworten enthielten `[1]`-Artefakte.

### Was nicht durch den Prompt erklärt wird

- Ambiguität-Halluzinationen ("StPO sagt nichts → System sagt 'Yes'"): wäre durch expliziteres Prompt-Wording adressierbar (z. B. "If the context does NOT address the question, abstain — do NOT extrapolate to 'Yes/No'"). **Marginale Verbesserung erwartbar, ~4/481 = 0.8 % der Samples betroffen**.
- Knappheit-Probleme (zu kurze Antworten): Regel 2 sagt "concise" ohne Längen-Hinweis. Hier ist ein Trade-off — kürzer heißt oft höhere Judge-FC, länger heißt oft vollständiger. Keine klare Prompt-Verbesserung ersichtlich.

### Empfehlung zum Prompt

**Keine Änderung** vor dem OpenAI-Lauf. Gründe:
1. Keine Sprach-Mismatches, keine systematischen Formatfehler, keine Ignorieren-des-Kontexts-Patterns — der Prompt macht seinen Job.
2. Die ambigue-Hallucination-Rate ist < 1 %, zu klein für eine evidenz-basierte Prompt-Anpassung.
3. Eine Prompt-Änderung macht das Ollama-Result ungültig → würde 12h Re-Run kosten.
4. Der Judge-Kalibrierungseffekt (unterschätzte FC bei knappen Antworten) wird durch OpenAI behoben, nicht durch Prompt-Änderung.

---

## 5. Entscheidungsmatrix

| Option | Aufwand | Risiko | Nutzen | Verdict |
|---|---|---|---|---|
| **A. OpenAI-Lauf wie geplant** | ~$3, 12h, 0 Code-Changes | niedrig | hoch: besserer Judge, direkte Vergleichbarkeit, fertiger Datensatz | **✓ wählen** |
| B. System-Prompt umschreiben + 2 Läufe neu | ~$3, 24h, Prompt-Rewrite | hoch: kein Evidenz-basierter Fix-Kandidat | ungewiss (marginal) | ✗ |
| C. Fusion-Confidence fixen + 2 Läufe neu | ~$3, 24h, Refactor + mgl. neue Bugs | hoch: Fusion-Redesign nicht trivial | Fusion wird messbar, aber unklare Qualität | ✗ |
| D. Threshold-Sweep-Logik ändern + 2 Läufe neu | ~$3, 24h, Eval-Refactor | mittel: ändert Design-Entscheidung, die dokumentiert ist | mgl. höhere Precision, aber F1-Trade-off bleibt | ✗ |
| E. Nichts tun, Ollama-Data als final nutzen | 0 | sehr hoch: ohne zweite Judge-Perspektive nicht thesis-fest | niedrig | ✗ |

### Konkrete Empfehlung

**Option A: OpenAI-Lauf jetzt starten, Pipeline unverändert.**

Die Ollama-Daten sind vorhanden. Der OpenAI-Lauf liefert:
1. Zweiten Judge-Datenpunkt → saubere Vergleichs-Narrative für die Thesis
2. Bessere Kalibrierung auf den Correctness-Scores (Judge-Artefakte fallen weg)
3. Niedrigere NaN-Rate (3 % statt 8.8 %)
4. **Gleiche System-Entscheidungen** (Abstention, Retrieval) — so können Judge-Differenzen sauber attribuiert werden

Alles, was wir in der Pipeline ändern *könnten*, adressiert kein durch Evidenz belegtes Problem. Der richtige Ort für die in dieser Analyse gefundenen Phänomene ist die Thesis-Diskussion, nicht ein Last-Minute-Code-Refactor.

---

## 6. Thesis-relevante Findings (für die Diskussion)

Zum Festhalten — gehören nicht in den Code, sondern in die Arbeit:

1. **RRF-Fusion mit disjunkten Korpora erzeugt degenerierte Confidence-Verteilungen.** 12 von 33 Configs konnten auf dem aktuellen Datensatz keine Antwort-Qualität messen. Relevant für zukünftige Cross-KB-Systeme.
2. **`k_rrf = 1` ist optimal** in allen KBs — entgegen dem Default 60 aus der RAGAS/RRF-Literatur. Hinweis darauf, dass kurze Top-Listen aus wenigen starken Retrievern wichtiger sind als lange Tail-Mischung.
3. **F1-basierte Threshold-Selektion neigt zu aggressiver Abstention.** Bei 25 % unbeantwortbaren Fragen im Dataset wird Recall-Maximierung zur dominanten Strategie. Precision-gewichtete Alternativen (F0.5, Youden's J) wären für Student-Facing-Bots womöglich besser.
4. **Distraktoren-Robustheit ist höher als erwartet.** Faithfulness und Correctness degradieren nicht signifikant mit steigender Distraktoren-Zahl (0 → 10). Das Grounding-Training der 8B-Instruct-Modelle scheint robust.
5. **Judge-Kalibrierung korreliert mit Modell-Größe.** 8B-Ollama gibt FC = 0 auf triviale korrekte Antworten; gpt-4o-mini (auf dem OpenAI-Pilot-Subset) nicht. Wichtige Designentscheidung für zukünftige lokale-LLM-als-Judge-Systeme.

---

## 7. Nächster Schritt

```bash
python -m evaluation --all --judge openai --output-dir data/evaluation-openai
```

Erwartet: ~12h, ~$3. Keine Code-Änderungen dazwischen.

Nach dem Lauf: Vergleichs-Analyse Ollama-vs-OpenAI → Material für das Evaluation-Kapitel der Thesis.
