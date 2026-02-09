from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import fitz

from pdf_extractor.config import (
    PDFExtractorRuntimeConfig,
    PDFExtractorSettings,
    get_pdf_extractor_config,
    get_pdf_extractor_settings,
)
from pdf_extractor.models import (
    DocumentContext,
    ExtractionResult,
    QualityFlag,
    SectionRecord,
    TableRecord,
)
from pdf_extractor.structure import extract_toc_entries, parse_sections
from pdf_extractor.tables import extract_tables


def _parse_german_date(value: str) -> date | None:
    value = value.strip().replace(",", " ").lower()
    # Simple numeric dd.mm.yyyy
    match = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", value)
    if match:
        d, m, y = map(int, match.groups())
        try:
            return date(y, m, d)
        except ValueError:
            pass
            
    # Month names
    months = {
        "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, 
        "mai": 5, "juni": 6, "juli": 7, "august": 8, "september": 9, 
        "oktober": 10, "november": 11, "dezember": 12
    }
    match = re.search(r"(\d{1,2})\.?\s*([a-züöä]+)\s*(\d{4})", value)
    if match:
        d_str, m_str, y_str = match.groups()
        month = months.get(m_str)
        if month:
            try:
                return date(int(y_str), month, int(d_str))
            except ValueError:
                pass
    return None


def _detect_metadata(
    page_texts: list[str],
    total_pages: int,
    source_file: Path,
    pdf_metadata: dict,
) -> tuple[DocumentContext, list[QualityFlag]]:
    flags: list[QualityFlag] = []
    first_pages_text = "\n".join(page_texts[:3]) # Scan first 3 pages
    
    # 1. Title
    title = (pdf_metadata.get("title") or "").strip()
    if not title or title.lower() in ("untitled", "o"):
        # Heuristic title extraction
        lines = [l.strip() for l in page_texts[0].splitlines() if l.strip()]
        # Look for typical Study Order titles
        candidates = [l for l in lines if "Ordnung" in l or "Studien" in l]
        title = candidates[0] if candidates else (lines[0] if lines else "Unknown Document")
    
    # 2. Institution
    institution = "Philipps-Universität Marburg"
    if "marburg" not in first_pages_text.lower():
         flags.append(QualityFlag(code="INSTITUTION_UNCERTAIN", message="Could not verify institution in text.", severity="info"))

    # 3. Version Date
    version_date: date | None = None
    # Look for "vom DD. Month YYYY"
    date_matches = re.finditer(r"(?:vom|Stand:?)\s+(.+?)(?:\n|$)", first_pages_text, re.IGNORECASE)
    for m in date_matches:
        parsed = _parse_german_date(m.group(1))
        if parsed:
            version_date = parsed
            break
            
    if not version_date:
        # Fallback: search any date
        dates = []
        for m in re.finditer(r"\d{1,2}\.\s*\d{1,2}\.\s*\d{4}", first_pages_text):
            parsed = _parse_german_date(m.group(0))
            if parsed: dates.append(parsed)
        if dates:
            version_date = max(dates)
        else:
            flags.append(QualityFlag(code="MISSING_DATE", message="No version date detected.", severity="warning"))
    
    # 4. Faculty
    faculty = None
    if "fb" in first_pages_text.lower() and "12" in first_pages_text:
        faculty = "FB12"
    elif "mathematik und informatik" in first_pages_text.lower():
        faculty = "FB12"
    else:
        flags.append(QualityFlag(code="MISSING_FACULTY", message="Faculty FB12 not explicitly detected.", severity="info"))

    context = DocumentContext(
        document_type="Studienordnung" if "ordnung" in title.lower() else "Document",
        title=title,
        institution=institution,
        version_date=version_date,
        faculty=faculty,
        total_pages=total_pages,
        source_language="de"
    )
    return context, flags


def _detect_toc_pages(page_texts: list[str]) -> set[int]:
    pages = set()
    for i, text in enumerate(page_texts):
        if "inhaltsverzeichnis" in text.lower():
            pages.add(i + 1)
            # Check next page for continuation?
            # Simple heuristic: if next page has many dotted lines or starts with Roman/Numbers
            # For now, let's stick to explicit TOC header or very dense structural lines.
            # But simplistic is better for stability.
    return pages


def resolve_source_file(
    request_source_file: str | None,
    document_id: str,
    settings: PDFExtractorSettings | None = None,
) -> Path:
    cfg = settings or get_pdf_extractor_settings()
    
    if request_source_file:
        candidate = Path(request_source_file)
        roots = [candidate, cfg.project_root / candidate, cfg.data_root_path / candidate, cfg.raw_dir_path / candidate]
        for r in roots:
            if r.exists(): return r.resolve()
        raise FileNotFoundError(f"Source file not found: {request_source_file}")

    # Default location
    default_path = cfg.raw_dir_path / f"{document_id}.pdf"
    if default_path.exists():
        return default_path.resolve()
        
    # Glob fallback
    files = list(cfg.raw_dir_path.glob("*.pdf"))
    if files:
        return sorted(files)[0].resolve()

    raise FileNotFoundError(f"No PDF found in {cfg.raw_dir_path}")


def run_extraction(
    document_id: str,
    source_file: Path,
    output_dir: Path | None = None,
    runtime_config: PDFExtractorRuntimeConfig | None = None,
    persist_result: bool = True,
) -> tuple[ExtractionResult, Path | None]:
    config = runtime_config or get_pdf_extractor_config()
    start_time = datetime.now(timezone.utc)
    
    # 1. Read PDF Text
    page_texts: list[str] = []
    pdf_metadata = {}
    with fitz.open(source_file) as doc:
        pdf_metadata = doc.metadata
        for page in doc:
            page_texts.append(page.get_text())

    # 2. Detect Metadata
    context, ctx_flags = _detect_metadata(page_texts, len(page_texts), source_file, pdf_metadata)

    # 3. Structure Analysis
    toc_pages = _detect_toc_pages(page_texts)
    toc_entries, toc_flags = extract_toc_entries(page_texts, toc_pages)
    sections, sec_flags = parse_sections(page_texts, toc_pages)

    # 4. Table Extraction
    tables, tbl_flags = extract_tables(source_file)

    # 5. Assign Tables to Sections
    # Simple heuristic assignment: table on page X belongs to section covering page X
    # If a section ends on page X, and table is on page X, does it belong to that section? Like yes.
    # What if multiple sections on page X? Assign to the one overlapping or nearest usage.
    # Simplified: Assign to section active at start of page X.
    
    # Pre-compute section page ranges
    # Actually, let's keep it simple: List tables in SectionRecord?
    # SectionRecord has `table_ids`.
    
    for table in tables:
        # Find candidate sections
        candidates = [s for s in sections if s.start_page <= table.page <= s.end_page]
        if candidates:
            # Pick the most specific one (highest logical depth: paragraph > part)
            # or just the last one defined?
            # Usually the one that *includes* the table physically.
            # But we don't have bounding boxes of sections easily from text.
            # Helper: Pick the section with smallest page range? (Most specific)
            best = min(candidates, key=lambda s: (s.end_page - s.start_page, -len(s.label)))
            table.section_id = best.section_id
            best.table_ids.append(table.table_id)
        else:
            tbl_flags.append(QualityFlag(code="ORPHAN_TABLE", message="Table not assigned to any section.", severity="info", context={"table_id": table.table_id}))

    result = ExtractionResult(
        document_id=document_id,
        source_file=str(source_file),
        context=context,
        toc_entries=toc_entries,
        sections=sections,
        tables=tables,
        quality_flags=[*ctx_flags, *toc_flags, *sec_flags, *tbl_flags],
        created_at=start_time
    )

    output_file: Path | None = None
    if persist_result:
        target_dir = output_dir or config.knowledgebases_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"{document_id}-{config.output_suffix}-{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        output_file = target_dir / fname
        
        output_file.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=config.json_indent),
            encoding="utf-8"
        )
        
        # Cleanup old if requested
        if not config.keep_previous_outputs:
            for f in target_dir.glob(f"{document_id}-{config.output_suffix}-*.json"):
                if f != output_file:
                    f.unlink(missing_ok=True)

    return result, output_file
