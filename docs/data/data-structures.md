# MARley Data Structures

This document defines every JSON file format produced and consumed by the MARley pipeline. For the Python data classes that represent these structures in code, see `docs/models/models.md`.

---

## 1. Extracted StPO (`stpo-extracted.json`)

Output of the PDF extractor. Contains the full StPO document decomposed into labelled sections, each with its plain text and any tables found on its pages.

### Structure

```json
{
  "source_file": "data/raw/msc-computer-science.pdf",
  "total_pages": 47,
  "sections": [
    {
      "section_id": "par-23",
      "label": "§23",
      "title": "Master's Thesis",
      "kind": "paragraph",
      "parent_section_id": "part-III",
      "start_page": 11,
      "end_page": 13,
      "text": "§23\nMaster's Thesis\n(1) The master's thesis ...",
      "tables": [
        {
          "table_id": "par-23-tbl-1",
          "page": 12,
          "headers": ["Column A", "Column B"],
          "rows": [["value 1", "value 2"]]
        }
      ]
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `source_file` | string | Path to the source PDF. |
| `total_pages` | integer | Total number of pages in the PDF. |
| `sections[]` | array | Ordered list of all detected sections. |
| `sections[].section_id` | string | Unique identifier. Format depends on kind (see below). |
| `sections[].label` | string | Display label as found in the PDF (e.g., `"§23"`, `"Appendix 2"`). |
| `sections[].title` | string | Section title extracted from the line following the label. |
| `sections[].kind` | string | One of `"preamble"`, `"toc"`, `"part"`, `"paragraph"`, `"appendix"`. |
| `sections[].parent_section_id` | string or null | ID of the containing part for paragraphs (e.g., `"part-III"`). `null` for top-level sections. |
| `sections[].start_page` | integer | First page of this section (1-indexed). |
| `sections[].end_page` | integer | Last page of this section (1-indexed). |
| `sections[].text` | string | Full plain text of the section, with normalized whitespace. |
| `sections[].tables` | array | Tables found on pages belonging to this section. Empty array if none. |
| `sections[].tables[].table_id` | string | Unique table ID. Format: `{section_id}-tbl-{n}`. |
| `sections[].tables[].page` | integer | Page where the table starts. |
| `sections[].tables[].headers` | array | Column header strings. |
| `sections[].tables[].rows` | array | Data rows, each an array of strings matching the header order. |

### Section ID Formats

| Kind | Format | Example |
|---|---|---|
| `preamble` | `preamble` | `preamble` |
| `toc` | `toc` | `toc` |
| `part` | `part-{ROMAN}` | `part-I`, `part-IV` |
| `paragraph` | `par-{NUMBER}` | `par-1`, `par-38` |
| `appendix` | `appendix-{NUMBER}` | `appendix-1`, `appendix-4` |

### Statistics (current extraction)

| Metric | Value |
|---|---|
| Total pages | 47 |
| Total sections | 48 |
| Total tables | 23 |
| Paragraphs (§1–§38) | 38 |
| Appendices (1–4) | 4 |
| Appendix 2 module rows | 54 (46 CS + 8 Conditional) |

---

## 2. FAQ Knowledge Bases (`faq-stpo.json`, `faq-ao.json`)

Input files for the FAQ chunker. Both FAQ sources share the same structure. Each file represents one knowledge base and contains a metadata header followed by an array of question-answer entries.

### Structure

```json
{
  "metadata": {
    "source": "faq-stpo",
    "version": "1.0",
    "created": "2026-03-05",
    "description": "Synthetic FAQ derived from the StPO of the M.Sc. Computer Science program."
  },
  "entries": [
    {
      "id": "stpo-0001",
      "question": "How many ECTS does the master thesis count for?",
      "answer": "The master thesis counts for 27 ECTS. Together with the disputation (3 ECTS), the final module comprises 30 ECTS in total.",
      "source": "§23 (2)"
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `metadata.source` | string | Identifier of the knowledge base. One of `"faq-stpo"` or `"faq-ao"`. |
| `metadata.version` | string | Version of the dataset. |
| `metadata.created` | string | ISO date of creation. |
| `metadata.description` | string | Human-readable description of the dataset. |
| `entries[].id` | string | Unique identifier. Format: `stpo-XXXX` for FAQ-StPO, `ao-XXXX` for FAQ-AO. |
| `entries[].question` | string | The question, written in English. |
| `entries[].answer` | string | The answer, written in English. Factually grounded in the source. |
| `entries[].source` | string | Provenance of the answer. For FAQ-StPO: section reference(s) from the StPO (e.g., `"§23 (2)"`). For FAQ-AO: origin such as `"advisory office"`, `"website"`, or a URL. |

### Notes

- **FAQ-StPO** entries are systematically derived from the StPO document. Each question must be answerable from the referenced section(s). The `source` field always references specific paragraphs or annexes of the StPO.
- **FAQ-AO** entries represent authentic student questions answered by the advisory office. The `source` field indicates where the answer originates (e.g., advisory office, department website). The FAQ-AO file will initially contain placeholder entries that are replaced with real data once available.
- Both FAQ files serve as **retrieval units** in the pipeline: each question-answer pair becomes one chunk during FAQ chunking.

---

## 3. Chunked StPO (`stpo-chunks.json`)

Output of the PDF chunker. Contains retrieval-ready chunks derived from the extracted StPO sections and tables.

### Structure

```json
{
  "source_file": "data/raw/msc-computer-science.pdf",
  "chunks": [
    {
      "chunk_id": "par-23-txt-1",
      "chunk_type": "text",
      "text": "III. Examination-related provisions > §23 Master's Thesis\n\n(1) The master's thesis ...",
      "token_count": 342,
      "metadata": {
        "document_id": "data/raw/msc-computer-science.pdf",
        "source_file": "data/raw/msc-computer-science.pdf",
        "section_id": "par-23",
        "section_kind": "paragraph",
        "section_label": "§23",
        "section_title": "Master's Thesis",
        "parent_section_id": "part-III",
        "heading_path": ["III. Examination-related provisions", "§23 Master's Thesis"],
        "start_page": 11,
        "end_page": 13,
        "chunk_index": 0,
        "table_id": null
      }
    }
  ],
  "stats": {
    "total_chunks": 135,
    "text_chunks": 100,
    "table_chunks": 35,
    "sections_processed": 48,
    "sections_skipped": 0,
    "tables_processed": 22,
    "min_tokens": 10,
    "median_tokens": 382,
    "max_tokens": 512,
    "total_tokens": 47783
  },
  "quality_flags": []
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `source_file` | string | Path to the source PDF. |
| `chunks[]` | array | Ordered list of all chunks. |
| `chunks[].chunk_id` | string | Unique identifier. Format: `{section_id}-txt-{n}` or `{section_id}-tbl-{table_id}-{n}`. |
| `chunks[].chunk_type` | string | One of `"text"` or `"table"`. |
| `chunks[].text` | string | The chunk content, including heading prefix and overlap. |
| `chunks[].token_count` | integer | Number of tokens (cl100k_base encoding). |
| `chunks[].metadata.document_id` | string | Source document identifier. |
| `chunks[].metadata.source_file` | string | Path to the source PDF. |
| `chunks[].metadata.section_id` | string | ID of the section this chunk belongs to. |
| `chunks[].metadata.section_kind` | string | Kind of the source section. |
| `chunks[].metadata.section_label` | string | Display label of the source section. |
| `chunks[].metadata.section_title` | string | Title of the source section. |
| `chunks[].metadata.parent_section_id` | string or null | ID of the parent section. |
| `chunks[].metadata.heading_path` | array | Breadcrumb path from root to section. |
| `chunks[].metadata.start_page` | integer | First page of the source section. |
| `chunks[].metadata.end_page` | integer | Last page of the source section. |
| `chunks[].metadata.chunk_index` | integer | 0-based index of this chunk within the section. |
| `chunks[].metadata.table_id` | string or null | Table ID for table chunks, null for text chunks. |
| `stats` | object | Aggregated statistics over all chunks. |
| `quality_flags[]` | array | Diagnostic flags (code, message, severity, context). |

### Chunk ID Formats

| Chunk type | Format | Example |
|---|---|---|
| Text | `{section_id}-txt-{n}` | `par-23-txt-1` |
| Table | `{section_id}-tbl-{table_id}-{n}` | `appendix-2-tbl-appendix-2-tbl-1-1` |

---

## 4. Chunked FAQs (`faq-stpo-chunks.json`, `faq-ao-chunks.json`)

Output of the FAQ chunker. Each FAQ entry becomes exactly one chunk with a combined question-answer text.

### Structure

```json
{
  "faq_source": "faq-stpo",
  "source_file": "data/knowledgebase/faq-stpo.json",
  "chunks": [
    {
      "chunk_id": "faq-stpo-stpo-0001",
      "chunk_type": "faq",
      "text": "Question: What do the Degree Program and Examination Regulations govern?\nAnswer: They supplement the General Regulations ...",
      "token_count": 52,
      "metadata": {
        "faq_source": "faq-stpo",
        "source_file": "data/knowledgebase/faq-stpo.json",
        "faq_id": "stpo-0001",
        "source_reference": "§1",
        "chunk_index": 0
      }
    }
  ],
  "stats": {
    "total_chunks": 999,
    "entries_total": 999,
    "entries_processed": 999,
    "entries_skipped": 0,
    "min_tokens": 16,
    "median_tokens": 37,
    "max_tokens": 416,
    "total_tokens": 58000
  },
  "quality_flags": []
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `faq_source` | string | Identifier of the FAQ source (`"faq-stpo"` or `"faq-ao"`). |
| `source_file` | string | Path to the input FAQ JSON file. |
| `chunks[]` | array | Ordered list of all FAQ chunks. |
| `chunks[].chunk_id` | string | Unique identifier. Format: `{faq_source}-{entry_id}`. |
| `chunks[].chunk_type` | string | Always `"faq"`. |
| `chunks[].text` | string | Combined text: `"Question: ...\nAnswer: ..."`. |
| `chunks[].token_count` | integer | Number of tokens (cl100k_base encoding). |
| `chunks[].metadata.faq_source` | string | FAQ source identifier. |
| `chunks[].metadata.source_file` | string | Path to the input FAQ JSON. |
| `chunks[].metadata.faq_id` | string | Original FAQ entry ID (e.g., `"stpo-0001"`). |
| `chunks[].metadata.source_reference` | string | Provenance of the answer (e.g., `"§23"`, `"advisory office"`). |
| `chunks[].metadata.chunk_index` | integer | Always 0 (one chunk per entry). |
| `stats` | object | Aggregated statistics over all chunks. |
| `quality_flags[]` | array | Diagnostic flags (code, message, severity, context). |

### Chunk ID Format

| Source | Format | Example |
|---|---|---|
| FAQ-StPO | `faq-stpo-{entry_id}` | `faq-stpo-stpo-0001` |
| FAQ-AO | `faq-ao-{entry_id}` | `faq-ao-ao-0001` |

---

## 5. Evaluation Dataset (`evaluation.json`)

The master evaluation dataset contains 100 questions used to evaluate the full MARley pipeline. These questions are **intentionally worded differently** from the FAQ entries to test genuine retrieval and generation quality rather than exact matching.

### Structure

```json
{
  "metadata": {
    "version": "1.0",
    "created": "2026-03-05",
    "description": "Evaluation dataset for MARley. Contains 100 questions stratified by category."
  },
  "questions": [
    {
      "id": "eval-001",
      "question": "What is the maximum duration allowed for completing the master thesis?",
      "reference_answer": "The master thesis must be completed within 6 months. An extension of up to 20% of the processing time is possible upon justified request.",
      "category": "direct",
      "relevant_chunks": [],
      "expected_abstention": false
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `metadata.version` | string | Version of the evaluation dataset. |
| `metadata.created` | string | ISO date of creation. |
| `metadata.description` | string | Human-readable description. |
| `questions[].id` | string | Unique identifier. Format: `eval-XXX`. |
| `questions[].question` | string | The evaluation question, in English. Deliberately worded differently from any FAQ entry. |
| `questions[].reference_answer` | string | The gold-standard reference answer. Empty string for unanswerable questions. |
| `questions[].category` | string | One of `"direct"`, `"multi-source"`, or `"unanswerable"`. |
| `questions[].relevant_chunks` | array | List of chunk IDs needed to answer this question. Empty in the master file; populated per knowledge base in the annotated variants (see Section 6). |
| `questions[].expected_abstention` | boolean | `true` if the system should abstain (unanswerable), `false` otherwise. |

### Category Definitions

| Category | Description | Example |
|---|---|---|
| `direct` | Answerable from a single chunk. | "What is the maximum duration for the master thesis?" |
| `multi-source` | Requires evidence from multiple sections or sources. | "Which modules can I take without any prerequisites?" |
| `unanswerable` | The knowledge base contains no sufficient evidence. | "What is the tuition fee for international students?" |

### Target Distribution

| Category | Count | Purpose |
|---|---|---|
| `direct` | ~40 | Test basic retrieval and factual generation |
| `multi-source` | ~35 | Test cross-section retrieval and synthesis |
| `unanswerable` | ~25 | Test abstention behavior |

---

## 6. Annotated Evaluation Datasets (`evaluation-*.json`)

Per-knowledge-base variants of the master evaluation dataset with `relevant_chunks` populated by manual annotation. Used for retrieval evaluation.

### Files

| File | Knowledge Base | Annotated Questions |
|---|---|---|
| `evaluation-stpo.json` | StPO chunks (142 text + table) | 75 |
| `evaluation-faq-stpo.json` | FAQ-StPO chunks (999 Q/A) | 75 |
| `evaluation-faq-ao.json` | FAQ-AO chunks (50 Q/A) | 21 |

### Structure

Identical to the master `evaluation.json`, with one addition in `metadata`:

```json
{
  "metadata": {
    "version": "1.0",
    "created": "2026-03-06",
    "description": "Evaluation dataset with relevant chunks for StPO.",
    "knowledge_base": "stpo"
  },
  "questions": [
    {
      "id": "eval-001",
      "question": "How long is the standard study period for the master's program?",
      "reference_answer": "The standard study period (Regelstudienzeit) is 4 semesters.",
      "category": "direct",
      "relevant_chunks": ["par-7-txt-1"],
      "expected_abstention": false
    }
  ]
}
```

### Additional Field

| Field | Type | Description |
|---|---|---|
| `metadata.knowledge_base` | string | Identifier of the knowledge base: `"stpo"`, `"faq-stpo"`, or `"faq-ao"`. |

### Annotation Criteria

- Only chunks that **directly contain the answer** are marked as relevant.
- For multi-source questions, all required chunks are listed.
- Unanswerable questions have empty `relevant_chunks` in all files.
- Chunk IDs reference the respective knowledge base's chunk format.

---

## 7. File Locations

```
data/
├── raw/
│   └── msc-computer-science.pdf          # Source PDF (StPO)
├── knowledgebase/
│   ├── stpo-extracted.json               # Extracted StPO (Section 1)
│   ├── faq-stpo.json                     # FAQ-StPO knowledge base (Section 2)
│   └── faq-ao.json                       # FAQ-AO knowledge base (Section 2)
├── chunks/
│   ├── stpo-chunks.json                  # Chunked StPO (Section 3)
│   ├── faq-stpo-chunks.json              # Chunked FAQ-StPO (Section 4)
│   └── faq-ao-chunks.json               # Chunked FAQ-AO (Section 4)
└── testing/
    ├── evaluation.json                   # Evaluation dataset, master (Section 5)
    ├── evaluation-stpo.json              # Annotated for StPO chunks (Section 6)
    ├── evaluation-faq-stpo.json          # Annotated for FAQ-StPO chunks (Section 6)
    └── evaluation-faq-ao.json            # Annotated for FAQ-AO chunks (Section 6)
```
