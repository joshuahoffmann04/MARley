# Generator

**Module:** `src/marley/generator/`
**Purpose:** Generate natural-language answers from retrieved context chunks using a locally hosted LLM.

The generator is the fourth stage of the MARley RAG pipeline. Given a student question and a set of context chunks (produced by the retrieval stage), it prompts an LLM to produce a grounded answer.

---

## Theoretical Background

Retrieval-Augmented Generation (RAG) was formalized by Lewis et al. (2020) as an approach that combines a non-parametric retrieval component with a parametric language model. Instead of relying solely on the model's internal knowledge (which may be outdated or incomplete), RAG conditions generation on externally retrieved evidence, improving factual accuracy and enabling knowledge updates without retraining.

In the RAG paradigm, the generator receives a query together with a set of retrieved context passages and produces an answer grounded in that context. The quality of generation depends on two factors: (1) the relevance and completeness of the retrieved context, and (2) the model's ability to synthesize information from the provided passages while avoiding hallucination.

MARley uses a **prompt-based generation** approach: retrieved chunks are formatted into a structured prompt that instructs the LLM to answer exclusively from the provided context. The prompt design follows established patterns for grounded question answering (Gao et al., 2024), including explicit instructions to abstain when context is insufficient (see [abstention.md](abstention.md)). The context chunks are numbered sequentially (`[1]`, `[2]`, ...) to provide a reference structure.

The system uses **Ollama** for local LLM inference, enabling fully offline operation and avoiding data privacy concerns associated with cloud-based APIs. The default model (`llama3.1:latest`) provides strong instruction-following capabilities suitable for the constrained generation task.

---

## Architecture

```
src/marley/generator/
├── __init__.py          # Exports: Generator, OllamaGenerator
├── base.py              # Re-export of Generator, GenerationResult from models
├── ollama.py            # Ollama LLM implementation
└── prompt.py            # Prompt templates and context formatting
```

### Generator Interface (`base.py`)

All generator implementations inherit from the abstract `Generator` class defined in `src/marley/models/generation.py`:

```python
class Generator(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the underlying model (e.g. 'llama3.1:latest')."""

    @abstractmethod
    def generate(self, query: str, context: list[dict[str, Any]]) -> GenerationResult:
        """Generate an answer given a query and context chunks."""
```

Each context dict follows the standard chunk format with `chunk_id`, `text`, and `metadata` keys.
The `model` property enables downstream code (e.g., the evaluation framework) to record which
model produced a given answer without inspecting `GenerationResult`.

### GenerationResult (`src/marley/models/generation.py`)

```python
@dataclass
class GenerationResult:
    answer: str                                          # Generated answer text
    model: str                                           # Model identifier (e.g., "llama3.1:latest")
    context_chunk_ids: list[str] = field(default_factory=list)  # Chunk IDs present in the context
    prompt_tokens: int = 0                               # Tokens in the prompt
    completion_tokens: int = 0                           # Tokens generated
```

---

## Prompt Design (`prompt.py`)

The prompt follows a system/user message structure:

**System prompt:** Instructs the LLM to act as a study advisor for the
M.Sc. Computer Science program at Philipps-Universität Marburg. Five
rules constrain the output:

1. Ground every answer strictly in the provided context passages.
2. Stay concise, precise, factually accurate.
3. Abstain with `ABSTENTION: <reason>` when the context is insufficient.
   An explicit clause covers the *silence-of-context* case: the model is
   told not to generalise from the absence of a prohibition to a "Yes"
   or from the absence of a permission to a "No" — a failure mode
   surfaced by the qualitative analysis.
4. No guessing, speculation, or outside knowledge.
5. Plain-text output, no `[1]` / chunk-ID references for the student.

**User message:** Contains the numbered context chunks followed by the question.

```
Context:
[1] The standard study period (Regelstudienzeit) is 4 semesters...
[2] The master thesis has 30 credits...

Question: How long is the standard study period?
```

### Context Formatting

**Function:** `format_context(chunks: list[dict[str, Any]]) → str`

Chunks are numbered sequentially (`[1]`, `[2]`, ...) to allow the LLM to reference specific passages. The numbering provides structure without implying relevance ranking. Returns `"No context provided."` for an empty chunk list.

**Function:** `build_messages(query: str, chunks: list[dict[str, Any]]) → list[dict[str, str]]`

Builds the complete message list for an LLM chat call. Returns a list with two dicts: a system message (using `SYSTEM_PROMPT`) and a user message containing the formatted context followed by the question.

---

## OllamaGenerator (`ollama.py`)

The default implementation uses the Ollama Python SDK to communicate with a locally hosted LLM.

**Default model:** `llama3.1:latest` — selected for its strong instruction-following capabilities and efficient local inference.

**Configuration:**

| Parameter | Default | Description |
|---|---|---|
| `model` | `"llama3.1:latest"` | Ollama model identifier |
| `base_url` | `"http://localhost:11434"` | Ollama server URL |

### Usage

```python
from src.marley.generator import OllamaGenerator
from src.marley.retrieval import BM25Retriever, load_chunks

# Setup
chunks = load_chunks("data/chunks/stpo-chunks.json")
retriever = BM25Retriever()
retriever.index(chunks)

# Retrieve + Generate
results = retriever.retrieve("How long is the study period?", k=5)
context = [{"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata} for r in results]

generator = OllamaGenerator()
answer = generator.generate("How long is the study period?", context)
print(answer.answer)
```

---

## Design Decisions

1. **Abstract interface:** The `Generator` base class allows swapping LLM backends without changing downstream code.

2. **Deterministic context formatting:** Chunks are always formatted the same way regardless of their source, ensuring consistent prompt structure.

3. **Token tracking:** `prompt_tokens` and `completion_tokens` are recorded for cost/performance analysis.

4. **No post-processing:** The generator returns the raw LLM output (stripped of whitespace). Any answer formatting or validation is left to the evaluation or frontend stages.
