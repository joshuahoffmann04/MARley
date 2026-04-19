# MARley Pre-Thesis Audit Plan

> Goal: Bring the repository to 100 % thesis-ready state before the final
> evaluation run and before writing the Bachelor thesis. Every finding is
> recorded, triaged, and either fixed or explicitly deferred with a reason.

**Date started:** 2026-04-19
**Author:** Joshua Hoffmann (with agent support)
**Scope:** All code, tests, evaluation, data, docs, infrastructure.

---

## Guiding principles

1. **Evidence before claims.** No phase is "done" without verification output
   (test run, grep result, doc build, etc.).
2. **Consistency across layers.** Code ↔ tests ↔ evaluation ↔ docs ↔ README
   must tell the *same* story. Discrepancies are the #1 audit target.
3. **No silent fixes.** Every change is logged in this file under
   "Findings & Resolutions" so it can be cited in the thesis if relevant.
4. **Parallel where independent, serial where shared.** Read-only audits run
   in parallel agents; edits are serialized by the main session.

---

## Stage 0 — Baseline Snapshot (serial, ~5 min)

Purpose: know what we're auditing before we audit it.

- [ ] `git status` — any uncommitted work? Any stray files?
- [ ] `git log --oneline -20` — last 20 commits sanity check.
- [ ] Run full unit-test suite once → capture pass/fail baseline.
- [ ] Run `evaluation/validate.py` once → confirm data integrity baseline.
- [ ] Record Python version, Torch/CUDA version, Ollama version, key package
      versions of the running environment (for reproducibility section of
      thesis).
- [ ] Record current coverage number from `.coverage` if valid, else ignore.

Deliverable: "Baseline" block filled in at the bottom of this file.

---

## Stage 1 — Code Audit (parallel, 6 agents)

Each agent owns one module. Read-only. Each agent reports in ≤200 words:
**(a) API/contract issues, (b) dead code / TODOs / FIXMEs, (c) error-handling
gaps, (d) obvious bugs, (e) doc-string coverage, (f) files touched & LOC.**

| Agent | Target directory              | Key files to inspect                       |
| ----- | ----------------------------- | ------------------------------------------ |
| A1    | `src/marley/extractor/`       | `extractor.py`                             |
| A2    | `src/marley/chunker/`         | `pdf_chunker.py`, `faq_chunker.py`         |
| A3    | `src/marley/retrieval/`       | `bm25.py`, `vector` impl, `hybrid.py`, `merged.py`, `base.py` |
| A4    | `src/marley/generator/` + `src/marley/abstention/` | `ollama.py`, `detection.py`, `confidence.py`, `pipeline.py` |
| A5    | `src/marley/server/`          | `app.py`, `service.py`, `pipeline.py`, `config.py`, `models.py`, `__main__.py` |
| A6    | `src/marley/models/` (Pydantic) | all `*.py`                               |

Each agent must also check:
- Imports are actually used (no stale imports).
- Type hints consistent.
- No `print()` left where a logger should be.
- Magic numbers justified or constants named.
- Consistency with the README architecture block.

Deliverable: one findings list per module, merged into "Findings & Resolutions".

---

## Stage 2 — Test Audit (parallel, 2 agents)

Purpose: verify tests actually test, not just pass.

**Agent T1 — unit tests (`tests/`).** Walk every `test_*.py`:
- Missing edge cases?
- Over-mocking that would hide real bugs (cf. memory on DB mocking)?
- Tests that skip silently?
- Test fixtures up to date with current models / configs?
- Any `xfail` / `skip` without reason?

**Agent T2 — evaluation tests (`evaluation/tests/`)**:
- Same questions applied to retrieval, generation, abstention, end-to-end
  evaluation tests.
- Also: do the evaluation tests actually exercise the CLI paths used in the
  real evaluation run?

Main session then runs:
- `pytest tests/ evaluation/tests/ -q --tb=short` → 0 failures required.
- Coverage target: no module < 80 % unless justified in findings.

Deliverable: testing section of "Findings & Resolutions".

---

## Stage 3 — Evaluation Audit (serial, main session + 1 agent)

Purpose: make sure the numbers that will end up in the thesis hold up.

Main session:
- Open `data/evaluation-ollama-1.0/ollama-evaluation.md` and
  `analysis-ollama.md`. For every number cited, confirm against the JSON
  result files (`e2e-results-*.json`, `retrieval-evaluation.json`,
  `rrf-tuning.json`, `abstention-evaluation.json`,
  `generation-evaluation*.json`).
- Spot-check: 33-configuration matrix really has 33 unique configs? List them.
- Spot-check: abstention F0.5 numbers match between code, tests, and doc.
- Spot-check: RRF tuning winner matches what the server actually uses.

**Agent E1** — statistical rigor review:
- Are seeds fixed? Is variance reported where it matters?
- Are judge-model prompts committed and versioned?
- Is the RAGAS judge config reproducible from the repo?
- Any leakage between tuning set and evaluation set?

Deliverable: "Evaluation consistency matrix" table (claim ↔ source) at the
bottom of this file.

---

## Stage 4 — Documentation Audit (parallel, 2 agents)

**Agent D1 — README.md + top-level docs.**
- Does every command in the README actually run on a fresh clone?
- Are all file paths mentioned in README still valid?
- Is the architecture block in sync with the code?
- Is the Ollama `OLLAMA_NUM_PARALLEL=2` setup clearly described (already
  done per recent commit — just verify)?
- License, author, citation block sane?

**Agent D2 — Sphinx-style docs in `docs/`.**
- `docs/source/**/*.md` — one per module. Up to date?
- `docs/testing/**/*.md` — unit-test docs aligned with actual tests?
- `docs/evaluation/**/*.md` and `docs/evaluation-testing/**/*.md` — aligned
  with evaluation code and test code?
- UML diagrams in `docs/uml/` — does `generate_uml.py` still produce valid
  output? Are generated diagrams checked in?
- Any orphan docs describing removed features?

Deliverable: list of outdated or missing doc sections → fix-list.

---

## Stage 5 — Infrastructure & Hygiene Audit (serial, main session)

- `pyproject.toml` — version, name, deps, Python version bound consistent
  with `requirements.txt` and README "Prerequisites"?
- `requirements.txt` — pinned appropriately for reproducibility?
- `.env.example` — covers every env var actually read by the code?
- `.gitignore` — does it exclude `.venv/`, `.chromadb-*`, `.coverage`,
  `__pycache__/`, `*.egg-info/`, any secret paths?
- `LICENSE` present and correct year/author?
- `bachelor/.txt` — what is this file? If it's a stray, delete.
- `data/evaluation-ollama-1.0/` — is this the canonical result folder? Is
  the naming (`-1.0`) intentional (implies a future `-1.1`)?
- Any `.pytest_cache`, `marley.egg-info/`, `.coverage` committed that
  shouldn't be?

Deliverable: infrastructure fix list.

---

## Stage 6 — Consolidation & Fixes (serial, main session)

Execute fixes from stages 1–5 in this order (lowest risk first):

1. Doc-only fixes (README, `docs/`).
2. Test-only fixes (new tests, renames, fixture updates).
3. Code fixes that don't change public API (dead code, type hints, comments).
4. Code fixes that do change behavior (must re-run relevant tests).
5. Infrastructure fixes (.gitignore, .env.example, etc.).
6. Final re-run: `pytest -q` **and** `evaluation/validate.py` **and** a
   smoke run of the server against the chat UI (manual).

Each fix batch = one git commit with a descriptive message. No
`git add -A`; stage by file.

---

## Stage 7 — Final Verification Gate (serial, main session)

No thesis-writing starts before **all** of these are green:

- [ ] `pytest -q` → 0 failures, 0 unexplained skips.
- [ ] `pytest --cov=src/marley --cov=evaluation` → coverage report saved.
- [ ] `python evaluation/validate.py` → exit 0.
- [ ] `python docs/uml/generate_uml.py` → fresh UML diagrams committed.
- [ ] Manual server smoke test: StPO + FAQ question → sensible answer, and
      out-of-scope question → abstention triggered.
- [ ] `git status` clean. All changes committed on `main` with clear
      messages.
- [ ] This AUDIT-PLAN.md has every checkbox ticked and every finding
      resolved or explicitly deferred.

Deliverable: "Verification" block at the bottom of this file, dated,
 with the git SHA of the final commit.

---

## Findings & Resolutions

_Populated as phases run. Template per finding:_

```
### F-<phase>-<nn> · <short title>
- **Severity:** blocker | major | minor | nit
- **Source:** <file:line or doc path>
- **Observation:** <what was seen>
- **Decision:** fix | defer | wontfix
- **Resolution commit / note:** <sha or reason>
```

<!-- Findings added below this line during execution. -->

### Stage 1 — Code Audit (results, 2026-04-19)

Six agents reported; every agent claim was spot-checked before recording
here. Where an agent erred, I say so explicitly and drop the finding.

#### Extractor (src/marley/extractor/ · ~637 LOC)

- **F-1-01 · E-01 · README shows `extract()` called without arguments** ·
  major. `extractor.py:580` requires `pdf_path` as a positional argument,
  but the README "Running the Evaluation" / extractor quick-start snippet
  (verify exact location in doc audit) allegedly calls it arg-less.
  *Decision:* fix docs in Stage 4/6 (not the function signature).
- **F-1-02 · E-02 · Missing empty-list guard** · minor.
  `extractor.py:275` does `all_page_numbers[-1]` without checking that
  the list is non-empty. Crashes on corrupted / empty PDF.
  *Decision:* fix — add early return.
- **F-1-03 · E-03 · `APPENDIX2_COL_INDICES = [0,3,4,7,8,9,10]` magic** ·
  minor. `extractor.py:31`. No comment explaining the 13-to-7 mapping.
  *Decision:* fix — add docstring/comment.
- **F-1-04 · E-04 · Type-hint inconsistency on public entry points** ·
  nit. `extract_page_texts(pdf_path: Path)` vs. `extract` / `save`
  accepting `str | Path`. *Decision:* unify to `str | Path`.
- Minor doc nits (E-05, E-07, E-08) merged into Stage 4 doc pass; E-06
  was a false positive (README ↔ code agreed).

#### Chunker (src/marley/chunker/ · ~919 LOC)

- **F-1-05 · C-01 · Asymmetric overlap-token bookkeeping** · major.
  `pdf_chunker.py:186` vs. `206-209`: main-window loop adds +1 for the
  joining space token, overlap accumulator does not. Real overlap
  shorter than target by ~N sentences. *Decision:* fix — make both
  branches count spaces the same way.
- **F-1-06 · C-04 · Inconsistent empty-chunk filtering** · minor.
  `_apply_heading_prefix()` drops empties; `_build_table_chunks()` does
  not (`pdf_chunker.py:433`). *Decision:* fix — filter consistently.
- **F-1-07 · C-06 · Hard-coded `chunk_index=0` in FAQ chunker** ·
  minor. `faq_chunker.py:253`. Documented as intentional, but only in
  `docs/source/chunker.md`, not in the dataclass. *Decision:* add inline
  docstring so a thesis reader sees the invariant in code.
- **F-1-08 · C-09 · Missing type guard on `source_file` in FAQ chunker** ·
  nit. `faq_chunker.py:202`. Silent string coercion of `None`.
  *Decision:* tighten type to `str | Path`.
- C-02 (set-iteration determinism): **false positive** — CPython 3.7+
  sets have not guaranteed ordering, but the codepath only calls
  `.add()`/`.contains()` and never iterates, so there is no risk.
  Dropped.
- C-05, C-08, C-10: pure doc nits; handled in Stage 4/6.

#### Retrieval (src/marley/retrieval/ + models/retrieval.py · ~662 LOC)

- **F-1-09 · R-01 · Default-`k` drift between base class and subclasses** ·
  minor. `base.py`/`models/retrieval.py:44` uses `DEFAULT_K`; all five
  concrete retrievers hard-code `k: int = 5`. If `DEFAULT_K` ever moves,
  subclasses silently ignore the change.
  *Decision:* fix — import and use `DEFAULT_K` in all five retrievers.
- **F-1-10 · R-02 · No error handling around ChromaDB calls** · minor.
  `vector.py:61-63, 91-93, 100-102`. Thesis-acceptable because the
  server layer has its own guards, but a clean message at the retriever
  level is nicer.
  *Decision:* defer (wontfix) unless trivial.
- **F-1-11 · R-03 · BM25 does not early-return on empty query** · nit.
  `bm25.py:56`. Library behavior undefined on `[]` tokens.
  *Decision:* fix — `if not query.strip(): return []`.
- **F-1-12 · R-06 · RRF tie-break is not stable** · minor.
  `models/retrieval.py:148-152`. When two docs share an RRF score, final
  order depends on Python's stable sort over insertion order of the
  accumulator dict. Deterministic in practice in CPython ≥ 3.7 but worth
  making explicit with a secondary key for thesis reproducibility.
  *Decision:* fix — add `chunk_id` as secondary sort key.
- **F-1-13 · R-09 · `FusionRetriever.index()` raises NotImplementedError** ·
  nit. Contract hole: sub-retrievers must be pre-indexed; this is only
  spelled out in the class docstring.
  *Decision:* defer — intentional, but add a validation check in
  `__init__` that every sub-retriever has `.size > 0`.

#### Generator + Abstention (src/marley/generator/ + abstention/ · ~200 LOC src + 197 scoring)

- **F-1-14 · A-01 · `compute_fusion_confidence` is NEVER called from the
  server pipeline (verified)** · **blocker** for thesis consistency.
  `server/pipeline.py:81` always does `compute_confidence(normalized)`,
  regardless of whether the retriever is a `FusionRetriever`. The
  evaluation pipeline (`evaluation/end_to_end/evaluate.py:135, 234`)
  does call `compute_fusion_confidence` for fusion. Therefore the
  production server computes confidence differently from the eval run.
  Thesis numbers for "Fusion" cannot be replicated by pointing a user at
  the running chat server. This is the single most important finding of
  Stage 1.
  *Decision:* fix — teach `run_with_abstention` to accept the sub-results
  (or a flag) and call `compute_fusion_confidence` when the retriever is
  a fusion retriever. Covered by tests in `tests/models/test_scoring.py`
  which already document the intended behavior.
- **F-1-15 · G-01 · `OllamaGenerator.generate` has no try/except** ·
  minor. `ollama.py:42`. Server-side 500 currently leaks details
  (see S-03). *Decision:* fix — wrap, raise a domain-specific exception.
- **F-1-16 · G-04 · No temperature / max_tokens configured** · minor.
  `ollama.py:42`. Ollama defaults are reasonable but not documented.
  *Decision:* fix — make temperature (e.g. 0.2) and `num_predict` (e.g.
  512) explicit constants, document them for the thesis.
- **A-02 · "F0.5 abstention" commit claim**: **false positive**. The
  agent only looked at `src/`. `evaluation/utils.py:198-214` implements
  F_beta(β=0.5); `evaluation/abstention/evaluate.py:251` and
  `evaluation/end_to_end/evaluate.py:79,167,168` select thresholds by
  F0.5; `evaluation/tests/test_utils.py:224-282` tests it. F0.5 is real.
  Dropped.
- Low-priority nits (A-03, A-04, A-05, A-06, A-07): doc-only; merge into
  Stage 4/6.

#### Server (src/marley/server/ · ~770 LOC)

- **F-1-17 · S-01 · `k_rrf_hybrid` parameter ignored at server layer** ·
  major. `service.py:74 + 87-90`. `_create_retriever` accepts the
  parameter but never passes it to `HybridRetriever(...)`. The empirical
  winner `k_rrf=1` (from `analysis-ollama.md`) is therefore uncustomisable
  through the server API.
  *Decision:* fix — pipe the parameter through.
- **F-1-18 · S-02 · `--mode` CLI flag is a no-op (verified)** ·
  major. `app.py:189-194` registers `--mode {all,chat,debug}`, parser
  assigns `args.mode`, and `args.mode` is **never read** (grep confirms
  zero uses in `src/`). Docs / README snippets showing
  `--mode chat` / `--mode debug` are misleading.
  *Decision:* fix — either implement per-mode route mounting or remove
  the flag and purge it from docs.
- **F-1-19 · S-03 · 500 handler leaks exception repr to the client** ·
  major. `app.py:153` builds
  `HTTPException(500, f"Pipeline error: {exc}")`. Thesis demo safety
  issue.
  *Decision:* fix — log details server-side, return a generic message.
- **F-1-20 · S-04 · Relative paths in `CHUNK_PATHS`** · minor.
  `config.py:24-28`. Server silently fails when launched from a
  directory other than the repo root.
  *Decision:* fix — resolve once at config construction relative to a
  known anchor (e.g., `Path(__file__).resolve().parents[3]`).
- S-05 (no CORS): **deferred** — single-origin deployment per thesis, so
  out of scope unless user requests.
- S-06, S-07, S-08, S-09: doc nits / polish; merged into Stage 4/6.

#### Pydantic models (src/marley/models/ · ~719 LOC incl. constants, scoring)

- **F-1-21 · M-03 · No range validation on `RetrievalResult.score`** ·
  nit. `retrieval.py:27`. After normalization scores should be in
  [0, 1]. `@dataclass` can't express this without a `__post_init__`.
  *Decision:* accept as-is (see M-01 decision below). Document the
  invariant in the dataclass docstring instead.
- **F-1-22 · M-09 · `QualityFlag.severity` is `str`** · nit.
  `quality.py:26`. Should be `Literal["info","warning","error"]`.
  *Decision:* fix — switch to `Literal`.
- M-01 (dataclasses vs. Pydantic): **design decision, not a bug**.
  FastAPI models in `server/models.py` already use Pydantic v2 where
  validation matters; the internal data carriers are intentionally
  `@dataclass`. A rewrite would churn every test for no runtime benefit
  before the final eval run. **Dropped.**
- M-02 ("`context_chunk_ids` camelCase outlier"): false positive — that
  name is snake_case. Dropped.
- M-04, M-05, M-07, M-08, M-10: low-value style nits; merged into
  Stage 4/6 docstring pass.

#### Cross-cutting Stage-1 summary

Blocker (must fix before thesis): **1** → F-1-14.
Major: **8** → F-1-01, F-1-05, F-1-17, F-1-18, F-1-19, F-1-09\*, F-1-20, F-1-06\*.
Minor / nit: ~14.
False positives explicitly dropped: 4 (C-02, E-06, A-02, M-01, M-02).

\* reclassified as minor after spot-check.

### Stage 2 — Test Audit (results, 2026-04-19)

Two agents reported. Baseline: 688 passed / 5 skipped.

#### The 5 skipped tests (verified)

All 5 use `pytest.mark.skipif` on integration *classes* with well-worded
reasons — not orphan skips:

- `TestBM25StPOIntegration` — StPO chunks not on disk (`test_bm25.py:73`)
- `TestBM25FAQStPOIntegration` — FAQ-StPO chunks not on disk (:98)
- `TestBM25FAQAOIntegration` — FAQ-AO chunks not on disk (:122)
- `TestOllamaGeneratorIntegration` — Ollama not running (`test_generator.py:230`)
- One of the vector/hybrid/merged integration classes under the same
  pattern.

All 5 skips are acceptable. No action required.

#### Blockers from Stage 2

- **F-2-01 · T-01 · No test locks the fusion-confidence behavior in the
  server pipeline.** `tests/server/test_service.py` builds fusion
  retrievers (line 134) but never asserts confidence semantics. Couples
  with F-1-14. *Decision:* fix the pipeline (F-1-14) **and** add a test
  that verifies `compute_fusion_confidence` is used for `FusionRetriever`.
- **F-2-02 · T-02 · Over-mocking of retriever factory in server tests.**
  `tests/server/test_service.py:38-46` and `test_api.py:32-42` patch
  retriever creation entirely. A refactor of `_create_retriever` would
  pass tests while breaking production. User explicitly called this out
  as a past pain point. *Decision:* fix — add at least one non-mocked
  test that exercises `_create_retriever("bm25", ["stpo"], "merged_pool")`
  against a tiny fixture corpus.
- **F-2-03 · T-03 · `--mode` flag untested.** Since the flag is a
  no-op (F-1-18), no test catches it. *Decision:* resolved when F-1-18 is
  fixed (either implement mode or delete it); add a test on the chosen
  outcome.

#### Majors from Stage 2

- **F-2-04 · T-04 · Weak confidence assertion** — `test_api.py:178-187`
  only checks `isinstance(confidence, float)`. *Decision:* fix —
  assert `0.0 <= c <= 1.0` and non-zero for non-abstaining responses.
- **F-2-05 · T-05 · Integration test tolerates wrong top-1.**
  `test_bm25.py:86-90` checks "par-23" in top-5, not top-1. *Decision:*
  tighten to top-1 where corpus guarantees it.
- **F-2-06 · T-06 · Missing edge-case tests.** `normalize_scores([])`,
  `retrieve(query, k=0)`, `k > corpus_size`, very-long query, no tests.
  *Decision:* fix — add a compact edge-case module in `tests/retrieval/`
  and `tests/models/test_scoring.py`.
- **F-2-07 · T-07 · No Unicode end-to-end retrieval test.** Agent flagged
  absence; I accept for thesis-readiness because the live corpus is
  German and umlauts are exercised implicitly by integration classes
  that are skipped in CI. *Decision:* downgrade to minor — add one
  non-skipped tiny German Unicode test.

#### Evaluation tests

- **F-2-08 · ET-01 · `evaluation/validate.py` has no unit tests** ·
  blocker for thesis-level completeness. Since this module drives the
  CLI "am I ready to run the eval" check, and the CLI itself will be
  cited in the thesis reproducibility section, it must be tested.
  *Decision:* fix — add `evaluation/tests/test_validate.py` covering
  missing chunk files, missing eval files, Ollama-required steps, and
  happy path.
- **F-2-09 · ET-02 · `evaluation/__main__.py` CLI has no tests** ·
  major. argparse dispatch, `--all`, `--retrieval`, `--generation`,
  `--abstention`, `--e2e`, `--judge` unverified. *Decision:* fix — add
  smoke tests that mock the work functions and verify dispatch + exit
  codes.
- **F-2-10 · ET-03 · End-to-end fusion confidence codepath untested** ·
  major. `evaluation/end_to_end/evaluate.py:135, 234` calls
  `compute_fusion_confidence`, but no e2e test exercises that branch.
  *Decision:* fix — add an e2e evaluation test that uses a mocked
  `FusionRetriever` populating `last_sub_results`.
- **F-2-11 · ET-04 · 33-configuration matrix test only checks count** ·
  minor. `test_config.py:66-68` verifies `len(CONFIGS) == 33` and the
  9 + 12 + 12 partition. *Decision:* fix — enumerate expected
  combinations programmatically and diff.
- ET-05, ET-06: minor; merge into F-2-06 edge-case test module.
- RAGAS mocking (ET-07) and fixture isolation (ET-08): **good**, no
  action.

#### Stage 2 summary

- Blockers: 4 (F-2-01, F-2-02, F-2-08, plus F-2-03 which resolves with
  F-1-18).
- Majors: 5 (F-2-04, F-2-05, F-2-06, F-2-09, F-2-10).
- Minor / nit: 3 (F-2-07, F-2-11, + rolled-up ET-05/ET-06).

### Stage 3 — Evaluation Audit (results, 2026-04-19)

One statistical-rigor agent + main-session spot-checks of JSON ↔ analysis.

#### Numbers match analysis-ollama.md ✓

Computed from raw JSON in `data/evaluation-ollama-1.0/`:

| claim (analysis) | value in analysis | recomputed from JSON | status |
| --- | --- | --- | --- |
| BM25 mean F1@5 | 0.187 | 0.1867 | ✓ |
| Vector mean F1@5 | 0.273 | 0.2726 | ✓ |
| Hybrid mean F1@5 | 0.259 | 0.2593 | ✓ |
| Hybrid best k_rrf on stpo, F1 | 1, 0.239 | 1, 0.239 | ✓ (agent) |
| merged-stpo+faq-stpo-hybrid FC | 0.63 | 0.634 | ✓ (agent) |
| 33 E2E configs | 33 | 33 files | ✓ |

No JSON ↔ analysis discrepancy. Methodology findings below are separate.

#### Methodological findings (P3 agent)

- **F-3-01 · P3-04 · Abstention threshold tuned AND evaluated on the
  same 100 queries** · **blocker for methodology section, not for code**.
  `evaluation/end_to_end/evaluate.py:79-168` sweeps Level-1 thresholds,
  picks the F0.5-maximum, then computes final abstention metrics on the
  identical query set. This is optimistic bias in the abstention
  precision / recall numbers cited in the thesis. The analysis doc's
  section 2.4 already observes that the sweep "neigt zu aggressiver
  Abstention" but does not call this bias by name.
  *Decision:* **do NOT silently fix the code before the OpenAI run**
  (that would change the numbers and force a re-run). Instead: add a
  one-paragraph note to `analysis-ollama.md` and to
  `docs/evaluation/abstention.md` describing the design (F0.5 sweep on
  the eval set) as an intentional operating-point selection, not a
  held-out generalization estimate. If a formal train/val/test split
  is desired, it is a post-thesis improvement.

- **F-3-02 · P3-01/P3-08 · Ollama judge model uses `:latest` tag, not a
  pinned digest** · major. `evaluation/judge.py:60` and
  `evaluation/__main__.py:521` default to `llama3.1:latest`. The current
  `ollama list` digest is not recorded anywhere. A re-run 3 months later
  might pull a different model.
  *Decision:* fix — record the actual digest in the README / analysis doc
  as "llama3.1 model file-SHA at time of run: <digest>"; optionally let
  the CLI accept a `--judge-model` flag (already exists) and document its
  use with a concrete tag.

- **F-3-03 · P3-07 · Pearson r(Abst-F1, FC) = 0.215 is cited but the
  computation is not in the repo** · minor. `analysis-ollama.md:89`
  cites the number with no accompanying script or notebook.
  *Decision:* fix — drop a small
  `data/evaluation-ollama-1.0/compute_stats.py` (or equivalent) that
  recomputes the correlation from the 33 E2E JSON files, so the
  reviewer can run it.

- **F-3-04 · P3-06 · Underpowered sub-configs with n_scored < 10** ·
  minor. Already flagged as a known characteristic (fusion and
  faq-ao configs produce n=0 by design). The analysis doc section
  2.1 discusses this explicitly.
  *Decision:* no code action — thesis table should report n for each
  cell so underpowered cells are self-evident.

- **F-3-05 · P3-02 · RAGAS-internal judge nondeterminism not
  controlled by a global seed** · minor. Each batch may produce
  slightly different scores on re-run.
  *Decision:* accept and disclose. Thesis variance disclosure (e.g.
  "~±0.02 on judge-based metrics from a 3×repeat spot-check") is
  sufficient.

- **P3-03 · "Judge prompts not committed"**: **downgraded**. RAGAS
  metric prompts are library-owned; the standard mitigation is to
  pin the RAGAS version in `requirements.txt` (already done) rather
  than copy-paste upstream prompts into the repo. **Dropped as a
  blocker.**

### Stage 4 — Documentation Audit (results, 2026-04-19)

Two agents. All findings below verified in main session before
recording.

#### README.md

- **F-4-01 · D1-01 · `extract()` snippet in Data Preparation shows
  call with no args** · blocker. README line 264 (the
  `python -c "..."` one-liner) passes no argument; the function
  needs `pdf_path`. Tying off F-1-01 — the code is fine, the doc is
  wrong.
  *Decision:* fix the README snippet.
- **F-4-02 · D1-02 / D1-04 · `chunk_stpo()` / `chunk_faq()` snippets
  pass wrong argument types** · blocker. README lines 269–270.
  `chunk_stpo` needs an `ExtractionResult`, `chunk_faq` needs a
  `FAQDataset`. README currently passes a file path string to both.
  *Decision:* rewrite those two lines to parse first, then chunk.
- **F-4-03 · D1-03 is a false positive** — the agent claimed
  `--mode` *is* implemented. Grep on `args.mode` returns zero
  matches across the repo. The flag is parsed and dropped. Merges
  with F-1-18. *Decision:* remove `--mode` from README snippets
  when the code is fixed (or remove it from the code if the README
  is what we want to keep — TBD in Stage 6).
- D1-05, D1-06, D1-08, D1-09: all confirmed **correct as-is**.
  No action.

#### docs/ (Sphinx-style)

- **F-4-04 · D2-01 · `k_rrf` default documented as 60 in multiple
  retrieval docs, actual code uses 1** · blocker.
  `docs/source/retrieval/overview.md`, `fusion.md`, and `hybrid.md`
  each state or exemplify `k_rrf=60` (the classic literature
  default). `src/marley/models/constants.py:16-32` sets
  `DEFAULT_K_RRF = DEFAULT_K_RRF_HYBRID = DEFAULT_K_RRF_FUSION = 1`
  from the the RRF tuning sweep sweep. Docs will contradict the thesis.
  *Decision:* fix all three doc files — change numbers to 1, cite
  `rrf-tuning.json` and the the RRF tuning sweep sweep.
- **F-4-05 · D2-02 · `docs/source/abstention.md` does not describe
  the fusion-confidence asymmetry between server and eval** ·
  major. Closely related to F-1-14.
  *Decision:* once F-1-14 is fixed (server calls
  `compute_fusion_confidence` for fusion), update this doc to
  state that server and eval now use the same path. If F-1-14 is
  NOT fixed, document the asymmetry explicitly instead.
- **F-4-06 · D2-03 · `docs/source/server.md` conflates Pydantic
  (used in `server/models.py`) with the dataclass-based internal
  models** · minor. Add one clarifying sentence.
- D2-04 (precision on `2/61 = 0.0328`): cosmetic; skip.

#### UML diagrams

- `docs/uml/marley_uml.puml` (+ .png) last touched 2026-03-24.
  `FusionRetriever` and `MergedRetriever` are in the puml file —
  *appears current*, but the `MergedRetriever` class was only added
  in commit 641c97d1 (2026-03-24 or later). Safe to regenerate in
  Stage 7 as part of the final verification gate. No new finding.

### Stage 3 + Stage 4 summary

- Blockers: 3 (F-3-01 methodology doc, F-4-01, F-4-02, F-4-04). Note
  F-3-01 is a documentation blocker — the code is intentional.
- Majors: 3 (F-3-02, F-4-05, + D1-03 rolling into F-1-18).
- Minor / nit: 4 (F-3-03, F-3-04, F-3-05, F-4-06).

### Stage 5 — Infrastructure & Hygiene (results, 2026-04-19)

- **F-5-01 · `.coverage` is tracked by git** · major. `git ls-files`
  confirms `.coverage` is in the index, despite being in `.gitignore`.
  It was added before the ignore entry existed. *Decision:* fix —
  `git rm --cached .coverage` in Stage 6 and commit.
- **F-5-02 · `.env.example` missing `OLLAMA_NUM_PARALLEL`** · minor.
  `evaluation/__main__.py:128` reads `OLLAMA_NUM_PARALLEL` but
  `.env.example` only documents `OPENAI_API_KEY`. *Decision:* fix —
  add it with explanatory comment.
- **F-5-03 · `bachelor/.txt` is a one-line stray** · nit. Contains
  only an Overleaf URL. *Decision:* rename to
  `bachelor/thesis-template.md` with a heading, or move the URL
  into the `bachelor/proposal/` README. Low priority.
- **F-5-04 · `data/evaluation-ollama-1.0/` is untracked** · major.
  For thesis reproducibility the JSON result files and
  `analysis-ollama.md` / `ollama-evaluation.md` must be committed.
  The `.chromadb-*` subdirectories inside must *not* be committed.
  The existing `.gitignore` catches `.chromadb-*/` globally, so a
  plain `git add data/evaluation-ollama-1.0/` should do the right
  thing. *Decision:* fix — track only the JSON + Markdown files.
- **F-5-05 · `Development Status :: 5 - Production/Stable`** in
  `pyproject.toml:13` · nit. Overclaim for a thesis artifact.
  *Decision:* fix — downgrade to
  `4 - Beta` (still honest, still ok for submission).
- **F-5-06 · `pyproject.toml` vs. `requirements.txt` split** · minor.
  `eval` + `dev` are optional in `pyproject.toml` but mandatory in
  `requirements.txt`. Fine in practice because the thesis run uses
  `requirements.txt`. No action needed unless the user wants to
  publish the package separately.

### Stage 5 summary

- Major: 2 (F-5-01, F-5-04).
- Minor / nit: 3 (F-5-02, F-5-03, F-5-05; F-5-06 is wontfix).

---

## Stage 6 — Fix Plan (risk-ascending, batched commits)

Rule: every fix that could change chunk boundaries, retrieval
ordering on tied scores, or generation output is **deferred** until
after the OpenAI evaluation run (the eval that produces the thesis
numbers). The items below are safe to apply before that run.

**Deferred to post-OpenAI-run:**
- F-1-05 (chunker overlap symmetry) — chunk-affecting.
- F-1-06 (`_build_table_chunks` empty filter) — chunk-affecting.
- F-1-12 (stable RRF tie-break) — retrieval-ordering-affecting.
- F-1-16 (generator temperature / num_predict) — generation-affecting.

**Batch 1 — infrastructure & hygiene** (F-5-01, F-5-02, F-5-04, F-5-05).

**Batch 2 — documentation fixes** (F-4-01, F-4-02, F-4-04, F-4-06,
F-3-01 threshold caveat note, F-3-02 model-digest note,
F-1-03 extractor column-index comment).

**Batch 3 — low-risk code fixes**
(F-0-01 validate CLI, F-1-02 extractor empty guard,
F-1-04 extractor type-hint unify, F-1-09 DEFAULT_K propagation,
F-1-11 BM25 empty-query guard, F-1-15 OllamaGenerator try/except,
F-1-17 pipe `k_rrf_hybrid`, F-1-19 generic 500 message,
F-1-20 absolute CHUNK_PATHS, F-1-22 QualityFlag Literal,
F-1-18 remove `--mode` (purely cosmetic)).

**Batch 4 — behavior change: fusion confidence in server pipeline**
(F-1-14). Server starts calling `compute_fusion_confidence` for
FusionRetriever. Pulls the server in line with the eval pipeline
that will be used for the final OpenAI run.

**Batch 5 — new & tightened tests** (F-2-01 lock fusion confidence,
F-2-02 non-mocked retriever integration, F-2-04 tighter confidence
assertions, F-2-06 edge cases, F-2-08 test_validate.py,
F-2-09 evaluation CLI tests, F-2-10 e2e fusion test,
F-2-11 enumerate 33-config matrix, F-2-07 tiny German Unicode
retrieval test).

**Batch 6 — Pearson-correlation reproducibility script** (F-3-03).

Each batch: one `git add` of named files (never `-A`), one commit
with a message describing the audit items resolved.

---

## Baseline

_Filled at end of Stage 0 (2026-04-19)._

- pytest (before audit): **688 passed, 5 skipped, 0 failed in 72.75 s**
- validate.py (before audit): **exit 0 — but see F-0-01**; it is a module
  with `validate_data_requirements()` and no `__main__` block, so running
  it as a script is a no-op. Thesis-readiness issue.
- Python: 3.12.2 — Torch: 2.5.1+cu121 — CUDA: 12.1 — GPU: NVIDIA RTX 4070 Ti SUPER
- Ollama: **not running during audit** (port 11434 refused). Only needed for
  the final evaluation run; unit tests and doc checks do not require it.
- Platform: Windows-11-10.0.26200-SP0
- Git HEAD at start of audit: `457d2732` on `main`, clean working tree apart
  from untracked `AUDIT-PLAN.md` and `data/evaluation-ollama-1.0/`.

### F-0-01 · `evaluation/validate.py` has no CLI entrypoint
- **Severity:** minor
- **Source:** `evaluation/validate.py`
- **Observation:** Exposes `validate_data_requirements()` but has no
  `if __name__ == "__main__"` block, so `python evaluation/validate.py`
  exits 0 silently. The AUDIT-PLAN referenced running it as a script.
- **Decision:** fix — add a minimal CLI that accepts `--steps` and exits
  non-zero on errors. Cheap and thesis-consistent.
- **Resolution commit / note:** _tbd in the fix stage_

---

## Verification

_Filled at end of Stage 7._

- pytest: …
- coverage: …
- validate.py: …
- UML regenerated: …
- manual server smoke: …
- Final git SHA: …
- Audit completed: …
