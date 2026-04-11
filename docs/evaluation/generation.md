# Generation Evaluation

> Evaluates answer generation quality using RAGAS metrics with controlled distractor testing.

## RAGAS Integration

Generation quality is measured via [RAGAS](https://docs.ragas.io/) 0.4.x (Retrieval-Augmented Generation Assessment), using a local Ollama LLM as the evaluator.

### Metrics

| Metric | RAGAS Class | What It Measures |
|---|---|---|
| **Faithfulness** | `Faithfulness` | Answer only uses information from the provided context |
| **Answer Relevancy** | `AnswerRelevancy` | Answer directly addresses the user's question |
| **Factual Correctness** | `FactualCorrectness` | Answer matches the reference answer |

All metrics produce scores in the range 0-1, where 1 is best.

### RAGAS Evaluation Pipeline

The RAGAS evaluation is performed in `_score_with_ragas()` (`evaluation/generation/evaluate.py`):

1. Create an `InstructorLLM` via `llm_factory()` using Ollama's OpenAI-compatible API (`/v1`) with `AsyncOpenAI`
2. Create `HuggingFaceEmbeddings` (all-mpnet-base-v2) for answer relevancy
3. Create RAGAS Collections metric instances: `Faithfulness`, `AnswerRelevancy`, `FactualCorrectness`
4. Score each metric in chunks via `_chunked_batch_score()` (batch_size=10) to avoid overwhelming the local Ollama server
5. Failed batches fall back to per-sample scoring with up to 3 retries; permanently failed samples receive NaN
6. Merge per-sample scores into a unified result dict

### Chunked Scoring and Error Handling

Ollama processes requests sequentially. Sending all samples at once causes massive retry storms.
`_chunked_batch_score()` splits the workload:

- **Batch size**: 10 samples per `batch_score()` call (`_RAGAS_BATCH_SIZE`)
- **Batch failure**: Falls back to per-sample `score()` calls
- **Per-sample retries**: Up to 3 attempts (`_SAMPLE_MAX_RETRIES`)
- **Final fallback**: `MetricResult(value=float("nan"))` for permanently failed samples
- **Aggregation**: `compute_generation_metrics()` excludes NaN values from averages

### RAGAS Input Schema

Each metric receives specific input fields:

| Metric | Input Fields |
|---|---|
| Faithfulness | `user_input`, `response`, `retrieved_contexts` |
| Answer Relevancy | `user_input`, `response` |
| Factual Correctness | `response`, `reference` |

### LLM Backend Configuration

| Parameter | Default | CLI Flag |
|---|---|---|
| `ollama_model` | `llama3.1:latest` | `--ollama-model` |
| `ollama_url` | `http://localhost:11434` | `--ollama-url` |

The evaluator LLM uses Ollama via its OpenAI-compatible API (`/v1`). Embeddings use `sentence-transformers/all-mpnet-base-v2` via RAGAS `HuggingFaceEmbeddings` (local, no Ollama needed).

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



## Output Files

| File | Content |
|---|---|
| `generation-evaluation.json` | Per-KB results with RAGAS scores per question x distractor level |
| `generation-evaluation-combined.json` | Combined-KB results |
