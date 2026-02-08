from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import fitz
import pdfplumber

from MARley.config import get_settings
from pdf_extractor.config import PDFExtractorRuntimeConfig
from pdf_extractor.models import (
    DocumentContext,
    ExtractionResult,
    QualityFlag,
    SectionRecord,
    TableRecord,
)
from pdf_extractor.structure import parse_sections

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


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


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


def _extract_context(
    page_texts: list[str],
    total_pages: int,
    source_file: Path,
    metadata: dict[str, str | None],
) -> tuple[DocumentContext, list[QualityFlag]]:
    flags: list[QualityFlag] = []
    first_pages_text = "\n".join(page_texts[:3])
    first_page_lines = [_normalize_whitespace(line) for line in page_texts[0].splitlines() if line.strip()] if page_texts else []

    title = _normalize_whitespace(metadata.get("title") or "")
    if not title:
        title_candidates = [line for line in first_page_lines if len(line) > 20]
        title = title_candidates[0] if title_candidates else ""

    document_type = "Studien- und Prüfungsordnung" if re.search(r"Studien-\s+und\s+Prüfungsordnung", first_pages_text, re.IGNORECASE) else None
    institution = "Philipps-Universität Marburg" if re.search(r"Philipps-Universität\s+Marburg", first_pages_text, re.IGNORECASE) else None

    version_date: date | None = None
    for pattern in (
        r"Fassung\s+vom\s+([^\n,;]+)",
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

    if version_date is None:
        filename_date = re.search(r"(\d{2})-(\d{2})-(\d{2})", source_file.stem)
        if filename_date:
            day = int(filename_date.group(1))
            month = int(filename_date.group(2))
            year = 2000 + int(filename_date.group(3))
            try:
                version_date = date(year, month, day)
                flags.append(
                    QualityFlag(
                        code="VERSION_DATE_FROM_FILENAME",
                        message="Versionsdatum konnte nicht im Text gefunden werden und wurde aus dem Dateinamen abgeleitet.",
                        severity="info",
                        context={"source_file": str(source_file)},
                    )
                )
            except ValueError:
                version_date = None

    faculty: str | None = None
    if re.search(r"\bFB\s*12\b", first_pages_text, re.IGNORECASE):
        faculty = "FB12"
    elif re.search(r"Fachbereich\s+Mathematik\s+und\s+Informatik", first_pages_text, re.IGNORECASE):
        faculty = "Fachbereich Mathematik und Informatik"

    if document_type is None:
        flags.append(
            QualityFlag(
                code="MISSING_DOCUMENT_TYPE",
                message="Dokumenttyp konnte nicht sicher erkannt werden.",
            )
        )
    if not title:
        flags.append(
            QualityFlag(
                code="MISSING_TITLE",
                message="Dokumenttitel konnte nicht sicher erkannt werden.",
            )
        )
    if institution is None:
        flags.append(
            QualityFlag(
                code="MISSING_INSTITUTION",
                message="Institution konnte nicht sicher erkannt werden.",
            )
        )
    if version_date is None:
        flags.append(
            QualityFlag(
                code="MISSING_VERSION_DATE",
                message="Versionsdatum konnte nicht sicher erkannt werden.",
            )
        )
    if faculty is None:
        flags.append(
            QualityFlag(
                code="MISSING_FACULTY",
                message="Fachbereich konnte nicht sicher erkannt werden.",
            )
        )

    context = DocumentContext(
        document_type=document_type,
        title=title or None,
        institution=institution,
        version_date=version_date,
        faculty=faculty,
        total_pages=total_pages,
    )
    return context, flags


def _clean_table_rows(rows: list[list[str | None]] | None) -> list[list[str]]:
    if not rows:
        return []
    cleaned: list[list[str]] = []
    for row in rows:
        cleaned.append([_normalize_whitespace(cell or "") for cell in row])
    return cleaned


def _extract_tables(source_file: Path) -> tuple[list[InterimTable], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    tables: list[InterimTable] = []
    counter = 1

    with pdfplumber.open(source_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                found_tables = page.find_tables() or []
            except Exception as exc:
                found_tables = []
                flags.append(
                    QualityFlag(
                        code="TABLE_DETECTION_FAILED",
                        message="Tabellenerkennung auf einer Seite ist fehlgeschlagen.",
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
                            message="Leere Tabelle wurde übersprungen.",
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
    sections: list[SectionRecord], tables: list[InterimTable]
) -> tuple[list[TableRecord], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    kind_rank = {"paragraph": 0, "annex": 1, "section": 2}
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
                    message="Tabelle konnte keinem Abschnitt zugeordnet werden.",
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
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def resolve_source_file(request_source_file: str | None, document_id: str) -> Path:
    settings = get_settings()

    if request_source_file:
        candidate = Path(request_source_file)
        candidate_paths = [candidate, settings.document_root / candidate, settings.raw_dir / candidate]
        for path in candidate_paths:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(f"Quell-PDF nicht gefunden: {request_source_file}")

    expected = settings.raw_dir / f"{document_id}.pdf"
    if expected.exists():
        return expected.resolve()

    alternatives = sorted(settings.raw_dir.glob("*.pdf"))
    if alternatives:
        return alternatives[0].resolve()

    raise FileNotFoundError(f"Keine PDF in {settings.raw_dir} gefunden.")


def run_extraction(
    document_id: str,
    source_file: Path,
    output_dir: Path,
    runtime_config: PDFExtractorRuntimeConfig,
) -> tuple[ExtractionResult, Path]:
    with fitz.open(source_file) as doc:
        metadata = doc.metadata or {}
        page_texts = [page.get_text("text") or "" for page in doc]
        total_pages = len(page_texts)

    context, context_flags = _extract_context(page_texts, total_pages, source_file, metadata)
    sections, structure_flags = parse_sections(page_texts)
    extracted_tables, table_flags = _extract_tables(source_file)
    tables, assignment_flags = _assign_tables_to_sections(sections, extracted_tables)

    result = ExtractionResult(
        document_id=document_id,
        source_file=str(source_file),
        context=context,
        sections=sections,
        tables=tables,
        quality_flags=[*context_flags, *structure_flags, *table_flags, *assignment_flags],
        created_at=datetime.now(timezone.utc),
    )

    output_file = _persist_result(
        result=result,
        output_dir=output_dir,
        runtime_config=runtime_config,
    )
    return result, output_file
