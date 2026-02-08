from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import fitz
import pdfplumber

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
from pdf_extractor.structure import extract_toc_entries, normalize_text, parse_sections

GERMAN_MONTHS: dict[str, int] = {
    "januar": 1,
    "februar": 2,
    "maerz": 3,
    "märz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}


@dataclass
class InterimTable:
    table_id: str
    page: int
    bbox: tuple[float, float, float, float] | None
    rows: list[list[str]]
    section_id: str | None = None
    flags: list[QualityFlag] = field(default_factory=list)


def _normalize_month_token(value: str) -> str:
    token = value.lower().strip()
    return token.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def _parse_date_token(value: str) -> date | None:
    token = value.strip().replace(",", " ")

    numeric = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", token)
    if numeric:
        day, month, year = [int(part) for part in numeric.groups()]
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            return None

    textual = re.search(r"(\d{1,2})\.\s*([A-Za-zÄÖÜäöüß]+)\s*(\d{4})", token)
    if textual:
        day = int(textual.group(1))
        month_token = textual.group(2)
        year = int(textual.group(3))
        month = GERMAN_MONTHS.get(month_token.lower())
        if month is None:
            month = GERMAN_MONTHS.get(_normalize_month_token(month_token))
        if month is None:
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def _detect_toc_pages(page_texts: list[str]) -> set[int]:
    toc_pages: set[int] = set()
    for page_no, text in enumerate(page_texts, start=1):
        lowered = text.lower()
        if "inhaltsverzeichnis" in lowered:
            toc_pages.add(page_no)
            continue

        # Some PDFs have continuation pages of the table of contents.
        if toc_pages and page_no == (max(toc_pages) + 1):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            candidate_lines = lines[:100]
            toc_like = sum(
                1
                for line in candidate_lines
                if re.match(r"^(§\s*\d+|[IVXLCDM]+\.\s*$|ANLAGE\s+\d+:?|\d+\s*$)", line, re.IGNORECASE)
            )
            prose_like = sum(
                1
                for line in candidate_lines
                if len(line) >= 70 and not re.match(r"^(§\s*\d+|[IVXLCDM]+\.\s*$|ANLAGE\s+\d+:?)", line, re.IGNORECASE)
            )
            # TOC continuation typically has many marker/page lines and very little long prose.
            if toc_like >= 12 and prose_like <= 2:
                toc_pages.add(page_no)
    return toc_pages


def _extract_context(
    page_texts: list[str],
    total_pages: int,
    source_file: Path,
    metadata: dict[str, str | None],
    first_page_headings: list[str],
) -> tuple[DocumentContext, list[QualityFlag]]:
    flags: list[QualityFlag] = []
    first_pages_text = "\n".join(page_texts[:4])
    first_page_lines = [line.strip() for line in page_texts[0].splitlines() if line.strip()] if page_texts else []

    metadata_title = (metadata.get("title") or "").strip()
    title: str | None = None
    if metadata_title and metadata_title.lower() not in {"o", "untitled"} and len(metadata_title) > 3:
        title = metadata_title
    elif first_page_headings:
        filtered = [
            line
            for line in first_page_headings
            if not re.match(r"^(vom|in der fassung vom)\b", line, re.IGNORECASE)
        ]
        title_candidate = " ".join(filtered).strip()
        title = title_candidate or None
    else:
        for line in first_page_lines:
            if "Studien-" in line and "Prüfungsordnung" in line:
                title = line
                break
        if title is None:
            title_candidates = [line for line in first_page_lines if len(line) > 20]
            title = title_candidates[0] if title_candidates else None

    document_type = "Studien- und Prüfungsordnung" if re.search(r"Studien-\s*und\s*Prüfungsordnung", first_pages_text, re.IGNORECASE) else None
    institution = "Philipps-Universität Marburg" if re.search(r"Philipps-Universität\s+Marburg", first_pages_text, re.IGNORECASE) else None

    version_date: date | None = None
    fassung_dates: list[date] = []
    for match in re.finditer(r"(?:in\s+der\s+)?Fassung\s+vom\s+([^\n,;]+)", first_pages_text, re.IGNORECASE):
        parsed = _parse_date_token(match.group(1))
        if parsed:
            fassung_dates.append(parsed)
    if fassung_dates:
        version_date = max(fassung_dates)

    if version_date is None:
        for pattern in (
            r"(\d{1,2}\.\s*[A-Za-zÄÖÜäöüß]+\s*\d{4})",
            r"(\d{1,2}\.\d{1,2}\.\d{2,4})",
        ):
            for match in re.finditer(pattern, first_pages_text, re.IGNORECASE):
                parsed = _parse_date_token(match.group(1))
                if parsed:
                    version_date = parsed
                    break
            if version_date:
                break

    faculty: str | None = None
    if re.search(r"\bFB\s*12\b", first_pages_text, re.IGNORECASE):
        faculty = "FB12"
    elif re.search(r"Fachbereichs?\s+Mathematik\s+und\s+Informatik", first_pages_text, re.IGNORECASE):
        faculty = "Fachbereich Mathematik und Informatik"

    if document_type is None:
        flags.append(QualityFlag(code="MISSING_DOCUMENT_TYPE", message="Document type could not be detected."))
    if title is None:
        flags.append(QualityFlag(code="MISSING_TITLE", message="Document title could not be detected."))
    if institution is None:
        flags.append(QualityFlag(code="MISSING_INSTITUTION", message="Institution could not be detected."))
    if version_date is None:
        flags.append(QualityFlag(code="MISSING_VERSION_DATE", message="Version date could not be detected."))
    if faculty is None:
        flags.append(QualityFlag(code="MISSING_FACULTY", message="Faculty could not be detected."))

    context = DocumentContext(
        document_type=document_type,
        title=title,
        institution=institution,
        version_date=version_date,
        faculty=faculty,
        total_pages=total_pages,
        source_language="de",
    )
    return context, flags


def _clean_table_rows(rows: list[list[str | None]] | None) -> list[list[str]]:
    if not rows:
        return []
    cleaned: list[list[str]] = []
    for row in rows:
        clean_row = [normalize_text(cell or "").replace("\n", " ") for cell in row]
        cleaned.append(clean_row)
    return cleaned


def _extract_tables(source_file: Path) -> tuple[list[InterimTable], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    tables: list[InterimTable] = []
    counter = 1

    with pdfplumber.open(source_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                found_tables = page.find_tables() or []
            except Exception as exc:  # pragma: no cover - pdf parsing edge case
                found_tables = []
                flags.append(
                    QualityFlag(
                        code="TABLE_DETECTION_FAILED",
                        message="Table detection failed for a page.",
                        severity="warning",
                        context={"page": page_number, "error": str(exc)},
                    )
                )

            if found_tables:
                extracted_data = [(table.extract(), tuple(float(v) for v in table.bbox)) for table in found_tables]
            else:
                extracted_data = [(_rows, None) for _rows in (page.extract_tables() or [])]

            for raw_rows, bbox in extracted_data:
                rows = _clean_table_rows(raw_rows)
                if not any(any(cell for cell in row) for row in rows):
                    flags.append(
                        QualityFlag(
                            code="EMPTY_TABLE_SKIPPED",
                            message="An empty table was skipped.",
                            severity="info",
                            context={"page": page_number},
                        )
                    )
                    continue

                table_id = f"tbl_{counter:04d}"
                counter += 1
                tables.append(
                    InterimTable(
                        table_id=table_id,
                        page=page_number,
                        bbox=bbox,
                        rows=rows,
                    )
                )

    return tables, flags


def _assign_tables_to_sections(
    sections: list[SectionRecord],
    tables: list[InterimTable],
) -> tuple[list[TableRecord], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    kind_rank = {"paragraph": 0, "annex": 1, "part": 2}
    table_records: list[TableRecord] = []

    for table in tables:
        candidates = [section for section in sections if section.start_page <= table.page <= section.end_page]
        if candidates:
            selected = min(
                candidates,
                key=lambda section: (
                    section.end_page - section.start_page,
                    kind_rank.get(section.kind, 99),
                    section.section_id,
                ),
            )
            table.section_id = selected.section_id
            selected.table_ids.append(table.table_id)
        else:
            flags.append(
                QualityFlag(
                    code="UNASSIGNED_TABLE",
                    message="Table could not be assigned to a section.",
                    severity="warning",
                    context={"table_id": table.table_id, "page": table.page},
                )
            )

        table_records.append(
            TableRecord(
                table_id=table.table_id,
                page=table.page,
                section_id=table.section_id,
                bbox=table.bbox,
                rows=table.rows,
                flags=table.flags,
            )
        )

    return table_records, flags


def _persist_result(
    result: ExtractionResult,
    output_dir: Path,
    runtime_config: PDFExtractorRuntimeConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{result.document_id}-{runtime_config.output_suffix}"

    if not runtime_config.keep_previous_outputs:
        for existing in output_dir.glob(f"{filename_prefix}-*.json"):
            existing.unlink(missing_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{filename_prefix}-{timestamp}.json"

    payload = result.model_dump(mode="json")
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=runtime_config.json_indent),
        encoding="utf-8",
    )
    return output_file


def resolve_source_file(
    request_source_file: str | None,
    document_id: str,
    settings: PDFExtractorSettings | None = None,
) -> Path:
    cfg = settings or get_pdf_extractor_settings()

    if request_source_file:
        candidate = Path(request_source_file)
        search_roots = [
            candidate,
            cfg.project_root / candidate,
            cfg.data_root_path / candidate,
            cfg.raw_dir_path / candidate,
        ]
        for path in search_roots:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(f"Source PDF not found: {request_source_file}")

    expected = cfg.raw_dir_path / f"{document_id}.pdf"
    if expected.exists():
        return expected.resolve()

    alternatives = sorted(cfg.raw_dir_path.glob("*.pdf"))
    if alternatives:
        return alternatives[0].resolve()

    raise FileNotFoundError(f"No PDF found in {cfg.raw_dir_path}")


def run_extraction(
    document_id: str,
    source_file: Path,
    output_dir: Path | None = None,
    runtime_config: PDFExtractorRuntimeConfig | None = None,
    persist_result: bool = True,
) -> tuple[ExtractionResult, Path | None]:
    config = runtime_config or get_pdf_extractor_config()
    target_output_dir = output_dir or config.knowledgebases_dir

    with fitz.open(source_file) as doc:
        metadata = doc.metadata or {}
        page_texts = [page.get_text("text") or "" for page in doc]
        total_pages = len(page_texts)
        first_page_headings: list[str] = []
        if total_pages > 0:
            page_dict = doc[0].get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        size = float(span.get("size", 0.0))
                        if text and size >= 13.0:
                            first_page_headings.append(text)

    context, context_flags = _extract_context(
        page_texts=page_texts,
        total_pages=total_pages,
        source_file=source_file,
        metadata=metadata,
        first_page_headings=first_page_headings,
    )
    toc_pages = _detect_toc_pages(page_texts)
    toc_entries, toc_flags = extract_toc_entries(page_texts, toc_pages)
    sections, structure_flags = parse_sections(page_texts, toc_pages)
    extracted_tables, table_flags = _extract_tables(source_file)
    tables, assignment_flags = _assign_tables_to_sections(sections, extracted_tables)

    result = ExtractionResult(
        document_id=document_id,
        source_file=str(source_file),
        context=context,
        toc_entries=toc_entries,
        sections=sections,
        tables=tables,
        quality_flags=[*context_flags, *toc_flags, *structure_flags, *table_flags, *assignment_flags],
        created_at=datetime.now(timezone.utc),
    )

    output_file: Path | None = None
    if persist_result:
        output_file = _persist_result(result=result, output_dir=target_output_dir, runtime_config=config)

    return result, output_file
