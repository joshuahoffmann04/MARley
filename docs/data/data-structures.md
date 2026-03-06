# MARley Data Structures

This document defines every JSON structure produced and consumed by the MARley pipeline.

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

## 2. FAQ Knowledge Base (`faq-stpo.json`, `faq-ao.json`)

Both FAQ sources share the same structure. Each file represents one knowledge base and contains a metadata header followed by an array of question-answer entries.

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

## 3. Evaluation Dataset (`evaluation.json`)

The evaluation dataset contains approximately 100 questions used to evaluate the full MARley pipeline. These questions are **intentionally worded differently** from the FAQ entries to test genuine retrieval and generation quality rather than exact matching.

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
| `questions[].relevant_chunks` | array | List of chunk IDs needed to answer this question. Populated after chunking is complete. |
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

## 4. File Locations

```
data/
├── raw/
│   └── msc-computer-science.pdf          # Source PDF (StPO)
├── knowledgebase/
│   ├── stpo-extracted.json               # Extracted StPO (48 sections, 23 tables)
│   ├── faq-stpo.json                     # FAQ-StPO knowledge base (999 entries)
│   └── faq-ao.json                       # FAQ-AO knowledge base (placeholder)
└── testing/
    └── evaluation.json                   # Evaluation dataset (100 questions)
```
