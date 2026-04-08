# LLM Judge

**Module:** `evaluation/judge/`
**Test files:** `evaluation/tests/judge/`

The LLM judge provides automated quality assessment for generated answers. It evaluates
each answer on three criteria — faithfulness, answer relevance, and correctness — by
sending a structured prompt to a language model and parsing its JSON response.

**See also:** [Generation Evaluation](generation.md) | [Evaluation Overview](overview.md)

---

## Theoretical Background

Automated evaluation of free-text answers is an open research problem. Traditional
n-gram metrics (ROUGE, BLEU) measure lexical overlap but miss semantic equivalence.
Learned metrics (BERTScore) improve semantic coverage but cannot assess factual grounding.

**LLM-as-judge** addresses this by using a language model to evaluate answers directly
(Zheng et al., 2023). The judge model reads the question, retrieved context, generated
answer, and reference answer, and outputs numeric quality scores. This approach:

- Captures semantic equivalence beyond n-gram overlap
- Can assess factual grounding (faithfulness) relative to the context
- Correlates better with human judgement than surface metrics
- Scales automatically without manual annotation

**Limitations:**
- Self-evaluation bias: using the same model as generator and judge inflates scores
- Positional and verbosity biases in some LLMs
- Results differ between model families

For the MARley evaluation, the first iteration uses the same Ollama model as the
generator. A second iteration with OpenAI (GPT-4o-mini) provides an independent judge
to mitigate self-evaluation bias.

---

## Architecture

```
evaluation/judge/
├── __init__.py          # Exports: Judge, JudgementResult, OllamaJudge, OpenAIJudge
├── base.py              # Judge ABC + JudgementResult dataclass
├── prompts.py           # System prompt + context formatter + message builder
├── ollama_judge.py      # OllamaJudge (local inference, no API key required)
└── openai_judge.py      # OpenAIJudge (requires OPENAI_API_KEY)
```

This mirrors the Generator architecture: an abstract base class with two concrete
backends, enabling seamless switching between Ollama and OpenAI.

---

## Judge Interface

```python
@dataclass
class JudgementResult:
    question_id: str
    faithfulness: float      # 0.0–1.0: answer grounded in context?
    answer_relevance: float  # 0.0–1.0: answer addresses the question?
    correctness: float       # 0.0–1.0: answer matches reference?
    model: str               # Judge model identifier

class Judge(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def judge(
        self,
        question_id: str,
        question: str,
        context: list[dict],
        generated_answer: str,
        reference_answer: str,
    ) -> JudgementResult: ...
```

---

## Evaluation Criteria

| Criterion | Description | Score 1.0 | Score 0.0 |
|---|---|---|---|
| **faithfulness** | Does the answer only use information present in the context? | Every claim is grounded in the context | Contains hallucinations (info not in context) |
| **answer_relevance** | Does the answer address the question? | Fully answers what was asked | Completely off-topic |
| **correctness** | Does the answer match the reference answer? | Factually correct and complete | Contradicts or ignores the reference |

All scores are in [0.0, 1.0]. Intermediate scores reflect partial compliance.

### Abstention Handling

If the generated answer is empty or starts with `ABSTENTION:`, the judge returns a
**sentinel result** without an LLM call:

| Criterion | Sentinel value | Rationale |
|---|---|---|
| faithfulness | 1.0 | No false claims were made |
| answer_relevance | 0.0 | Question was not answered |
| correctness | 0.0 | Question was not answered |

---

## Prompt Design

The judge uses a **single structured prompt** requesting all three scores in one call,
reducing latency compared to three separate calls.

**System prompt:** Instructs the model to output exactly one JSON object:
```json
{"faithfulness": <float>, "answer_relevance": <float>, "correctness": <float>}
```

**User message:** Contains the question, numbered context passages, generated answer,
and reference answer.

The `format="json"` parameter is used with Ollama to enforce JSON output. For OpenAI,
`response_format={"type": "json_object"}` achieves the same effect.

### JSON Parsing

Scores are parsed with a two-stage fallback:
1. **Strict JSON parsing** via `json.loads()`.
2. **Regex extraction** of the first `{...}` object if strict parsing fails.
3. **Zero default** for missing or non-numeric keys.
4. **Clamping** to [0.0, 1.0] for out-of-range values.

---

## OllamaJudge

Uses a locally hosted Ollama model — no API key required. The model identifier
and server URL are configurable.

```python
from evaluation.judge import OllamaJudge

judge = OllamaJudge(model="llama3.1:latest")  # default
result = judge.judge(
    question_id="eval-001",
    question="How long is the standard study period?",
    context=[{"chunk_id": "c1", "text": "The standard study period is 4 semesters."}],
    generated_answer="The study period is 4 semesters.",
    reference_answer="4 semesters.",
)
print(result.faithfulness, result.answer_relevance, result.correctness)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"llama3.1:latest"` | Ollama model identifier |
| `base_url` | `"http://localhost:11434"` | Ollama server URL |

---

## OpenAIJudge

Uses the OpenAI API for independent judgement. Requires the `openai` package and an
API key (via constructor or `OPENAI_API_KEY` environment variable).

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from evaluation.judge import OpenAIJudge

judge = OpenAIJudge(model="gpt-4o-mini")  # default
result = judge.judge(
    question_id="eval-001",
    question="How long is the standard study period?",
    context=[{"chunk_id": "c1", "text": "The standard study period is 4 semesters."}],
    generated_answer="The study period is 4 semesters.",
    reference_answer="4 semesters.",
)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gpt-4o-mini"` | OpenAI model identifier |
| `api_key` | `None` → env var | OpenAI API key |

---

## Integration with Generation Evaluation

Pass a judge to `run_generation_evaluation()` or `run_and_report()`:

```python
from evaluation.generation.evaluate import run_and_report
from evaluation.judge import OllamaJudge
from src.marley.generator import OllamaGenerator
from src.marley.retrieval import load_chunks

chunks = load_chunks("data/chunks/stpo-chunks.json")
generator = OllamaGenerator()
judge = OllamaJudge()

report = run_and_report(
    generator, chunks,
    "data/testing/evaluation-stpo.json",
    distractor_levels=[0, 5, 10],
    knowledge_base="stpo",
    judge=judge,
)
# report["metrics"] contains avg_faithfulness, avg_answer_relevance, avg_correctness
```

Without a judge, the three LLM score fields default to 0.0 in the report.
