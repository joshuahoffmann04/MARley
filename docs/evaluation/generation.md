# Generation Evaluation

> Evaluates answer generation quality using RAGAS metrics with controlled distractor testing.

## RAGAS Integration

Generation quality is measured via [RAGAS](https://docs.ragas.io/) 0.4.x
(Retrieval-Augmented Generation Assessment). The LLM that RAGAS uses as a
judge is configurable at the CLI level; the generator under evaluation is
always the Ollama-served chat model.

### Metrics

| Metric | RAGAS Class | What It Measures |
|---|---|---|
| **Faithfulness** | `Faithfulness` | Answer only uses information from the provided context |
| **Answer Relevancy** | `AnswerRelevancy` | Answer directly addresses the user's question |
| **Factual Correctness** | `FactualCorrectness` | Answer matches the reference answer |

All metrics produce scores in the range 0-1, where 1 is best.

### Scoring

The judge is built in `evaluation/judge.py::make_judge()` and passed down
as a single `Judge` object to `_score_with_ragas()`. The factory accepts
one of two backends:

| Backend | Judge LLM | Client | Batch size | Notes |
|---|---|---|---|---|
| `ollama` (default) | `--ollama-model` via Ollama's OpenAI-compatible API | `AsyncOpenAI(base_url=<ollama>/v1)` | 20 | Assumes `OLLAMA_NUM_PARALLEL=2` |
| `openai` | `gpt-4o-mini` | `AsyncOpenAI(api_key=<OPENAI_API_KEY>)` | 50 | Requires `.env` with the key |

Both backends use the same embedding model for Answer Relevancy:
`sentence-transformers/all-mpnet-base-v2`, loaded on the GPU via
`HuggingFaceEmbeddings(..., device="cuda")`.

`_score_with_ragas()` then runs each RAGAS collection metric
(`Faithfulness`, `AnswerRelevancy`, `FactualCorrectness`) through
`_chunked_batch_score()`, which splits the sample list into chunks of
`judge.batch_size`. The batch size is tuned per backend: Ollama's
2-parallel serving stalls on bigger queues, while OpenAI is rate-limit
bound and benefits from the larger window.

### Failure handling

`_chunked_batch_score()` is resilient against transient judge failures:

- **Batch failure**: Falls back to per-sample `score()` calls for the chunk.
- **Per-sample retries**: Up to 3 attempts (`_SAMPLE_MAX_RETRIES`).
- **Final fallback**: `MetricResult(value=float("nan"))` for permanently failed samples.
- **Aggregation**: `compute_generation_metrics()` excludes NaN values from averages.

### RAGAS Input Schema

Each metric receives specific input fields:

| Metric | Input Fields |
|---|---|
| Faithfulness | `user_input`, `response`, `retrieved_contexts` |
| Answer Relevancy | `user_input`, `response` |
| Factual Correctness | `response`, `reference` |

### CLI flags that control the judge

| Flag | Default | Description |
|---|---|---|
| `--judge` | `ollama` | Judge backend (`ollama` or `openai`) |
| `--ollama-model` | `llama3.1:latest` | Ollama model used as judge (only when `--judge ollama`) |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL (always used for the generator; also for the Ollama judge) |

`--ollama-model` never controls the OpenAI judge; the OpenAI backend is
hard-wired to `gpt-4o-mini`. The generator model is also always the
Ollama model.

The same judge factory is used by the E2E evaluation, which scores
every non-abstained answerable answer with these three metrics. See
[end-to-end.md](end-to-end.md#answer-quality-scoring) for the
scoring-scope matrix and how the metrics are aggregated per E2E
configuration.

## Distractor Testing

Generation evaluation uses a controlled methodology to test robustness against retrieval noise.

### Methodology

For each answerable question:

1. **Relevant chunks**: Ground-truth chunks from the evaluation dataset
2. **Distractor selection**: Non-relevant chunks ranked by BM25 similarity to the question (hardest distractors first)
3. **Context assembly**: Relevant chunks + N distractors, shuffled with a deterministic seed
4. **Generation**: LLM generates an answer from the assembled context
5. **Scoring**: RAGAS evaluates the generated answer

### Distractor Levels

By default, each question is evaluated at 11 distractor levels: 0, 1, 2, ..., 10.

- **Level 0**: Only ground-truth relevant chunks (ideal retrieval)
- **Level 10**: Ground-truth chunks + 10 BM25-ranked distractors (noisy retrieval)

This reveals how generation quality degrades as retrieval noise increases.

### Distractor Selection (`select_distractors()`)

- Indexes only non-relevant chunks into a temporary BM25 retriever
- Retrieves the top-`max_distractors` by query similarity
- Produces the **hardest** (most confusing) distractors first

### Context Assembly (`_assemble_context()`)

- Combines relevant chunks with the selected distractors
- Shuffles using a fixed seed per question (`hash(q["id"]) & 0xFFFFFFFF + n_distractors`)
- Deterministic but unpredictable ordering for the LLM

## Data Classes

### GenerationEvalResult

Per-question, per-distractor-level result:

| Field | Type | Description |
|---|---|---|
| `question_id` | `str` | Question identifier |
| `num_distractors` | `int` | Number of distractor chunks in context |
| `generated_answer` | `str` | LLM-generated answer |
| `reference_answer` | `str` | Ground-truth reference answer |
| `context_chunk_ids` | `list[str]` | Chunk IDs in the assembled context |
| `faithfulness` | `float` | RAGAS faithfulness score (0-1, NaN on failure) |
| `answer_relevance` | `float` | RAGAS answer relevancy score (0-1, NaN on failure) |
| `correctness` | `float` | RAGAS factual correctness score (0-1, NaN on failure) |

### GenerationMetrics

Aggregated metrics over all results (NaN values excluded from averages):

| Field | Type | Description |
|---|---|---|
| `num_results` | `int` | Total number of results (including NaN) |
| `num_queries` | `int` | Number of unique questions |
| `results_by_distractors` | `dict[int, int]` | Count of results per distractor level |
| `avg_faithfulness` | `float` | Mean faithfulness (NaN excluded) |
| `avg_answer_relevance` | `float` | Mean answer relevancy (NaN excluded) |
| `avg_correctness` | `float` | Mean factual correctness (NaN excluded) |

Implementation: `evaluation/generation/metrics.py` (`compute_generation_metrics()`)

## Evaluation Modes

### Single-KB Evaluation

Evaluates generation for each KB independently.

**Process**: Load corpus -> Load questions -> Run generation evaluation -> Score with RAGAS -> Aggregate metrics

### Combined-KB Evaluation

Evaluates generation with merged multi-KB context.

**Process**:
1. Merge chunks from all KBs (`merge_chunks()`)
2. Merge evaluation data (`merge_evaluation_data()`)
3. Run standard generation evaluation on the merged data

The merged corpus provides harder distractors from a larger pool, testing robustness in a realistic multi-KB scenario.

Implementation: `evaluation/generation/combined.py`

## CLI Usage

```bash
# Default Ollama judge over the full eval set
python -m evaluation --generation

# OpenAI judge (requires OPENAI_API_KEY in .env)
python -m evaluation --generation --judge openai

# Quick subset: 10 questions × distractor levels {0, 5, 10} on stpo only
python -m evaluation --generation --subset 10 --distractor-levels 0,5,10 \
    --judge openai --kb-filter stpo --output-dir data/evaluation/subset-openai
```

| Flag | Default | Purpose |
|---|---|---|
| `--judge {ollama,openai}` | `ollama` | Judge backend (generator is always Ollama) |
| `--subset N` | `None` | Limit the eval to the first N questions per KB |
| `--distractor-levels 0,5,10` | `0..10` | Comma-separated distractor counts |
| `--kb-filter {stpo,faq-stpo,faq-ao}` | `None` | Restrict to a single KB; skips the combined-KB run |
| `--ollama-model` | `llama3.1:latest` | Generator model (and Ollama judge model) |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |

The combined-KB run is automatically skipped when `--subset` or
`--kb-filter` is active — those are verification shortcuts, not
production reports.

## Output Files

| File | Content |
|---|---|
| `generation-evaluation.json` | Per-KB results with RAGAS scores per question x distractor level |
| `generation-evaluation-combined.json` | Combined-KB results |
