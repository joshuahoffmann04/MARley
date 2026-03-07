# Generator

**Module:** `src/marley/generator/`
**Purpose:** Generate natural-language answers from retrieved context chunks using a locally hosted LLM.

The generator is the fourth stage of the MARley RAG pipeline. Given a student question and a set of context chunks (produced by the retrieval stage), it prompts an LLM to produce a grounded answer.

---

## Architecture

```
src/marley/generator/
├── __init__.py          # Exports: Generator, OllamaGenerator
├── base.py              # Abstract Generator interface
├── ollama.py            # Ollama LLM implementation
└── prompt.py            # Prompt templates and context formatting
```

### Generator Interface (`base.py`)

All generator implementations inherit from the abstract `Generator` class:

```python
class Generator(ABC):
    def generate(self, query: str, context: list[dict]) -> GenerationResult:
        """Generate an answer given a query and context chunks."""
```

Each context dict follows the standard chunk format with `chunk_id`, `text`, and `metadata` keys.

### GenerationResult (`src/marley/models/generation.py`)

```python
@dataclass
class GenerationResult:
    answer: str                        # Generated answer text
    model: str                         # Model identifier (e.g., "llama3.1:latest")
    context_chunk_ids: list[str]       # Chunk IDs present in the context
    prompt_tokens: int                 # Tokens in the prompt
    completion_tokens: int             # Tokens generated
```

---

## Prompt Design (`prompt.py`)

The prompt follows a system/user message structure:

**System prompt:** Instructs the LLM to act as a study advisor for the M.Sc. Computer Science program at Philipps-Universität Marburg. The LLM is directed to answer based ONLY on the provided context and to state clearly when the context is insufficient.

**User message:** Contains the numbered context chunks followed by the question.

```
Context:
[1] The standard study period (Regelstudienzeit) is 4 semesters...
[2] The master thesis has 30 credits...

Question: How long is the standard study period?
```

### Context Formatting

Chunks are numbered sequentially (`[1]`, `[2]`, ...) to allow the LLM to reference specific passages. The numbering provides structure without implying relevance ranking.

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

1. **Abstract interface:** The `Generator` base class allows swapping LLM backends (e.g., OpenAI, Anthropic) without changing downstream code.

2. **Deterministic context formatting:** Chunks are always formatted the same way regardless of their source, ensuring consistent prompt structure.

3. **Token tracking:** `prompt_tokens` and `completion_tokens` are recorded for cost/performance analysis.

4. **No post-processing:** The generator returns the raw LLM output (stripped of whitespace). Any answer formatting or validation is left to the evaluation or frontend stages.
