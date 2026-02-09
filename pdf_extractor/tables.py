from __future__ import annotations

from pathlib import Path

import pdfplumber
from fitz import Rect

from pdf_extractor.models import QualityFlag, TableRecord
from pdf_extractor.structure import normalize_whitespace


def _clean_table_rows(rows: list[list[str | None]] | None) -> list[list[str]]:
    if not rows:
        return []
    cleaned: list[list[str]] = []
    for row in rows:
        clean_row = [normalize_whitespace(cell or "") for cell in row]
        cleaned.append(clean_row)
    return cleaned


def extract_tables(source_file: Path) -> tuple[list[TableRecord], list[QualityFlag]]:
    flags: list[QualityFlag] = []
    tables: list[TableRecord] = []
    counter = 1

    try:
        with pdfplumber.open(source_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    # pdfplumber find_tables is heuristic but often good
                    found_tables = page.find_tables() or []
                except Exception as exc:
                    flags.append(
                        QualityFlag(
                            code="TABLE_DETECTION_ERROR",
                            message=f"Table detection failed on page {page_number}",
                            severity="warning",
                            context={"error": str(exc)},
                        )
                    )
                    found_tables = []

                if found_tables:
                    extracted_data = [(table.extract(), tuple(float(v) for v in table.bbox)) for table in found_tables]
                else:
                    # Fallback to simple grid extraction if no explicit tables found?
                    # Actually, extract_tables() without arguments tries to find specific intersection based tables.
                    # If find_tables() failed or found nothing, extract_tables() is usually similar but strict.
                    # Let's trust find_tables() mostly.
                    # But if we want to be safe:
                    raw_data = page.extract_tables()
                    extracted_data = [(rows, None) for rows in raw_data] if raw_data else []

                for raw_rows, bbox in extracted_data:
                    cleaned_rows = _clean_table_rows(raw_rows)
                    # Skip empty tables
                    if not any(any(cell for cell in row) for row in cleaned_rows):
                        continue

                    table_id = f"tbl_{counter:04d}"
                    counter += 1
                    
                    tables.append(
                        TableRecord(
                            table_id=table_id,
                            page=page_number,
                            bbox=bbox,
                            rows=cleaned_rows,
                            flags=[],
                        )
                    )

    except Exception as exc:
         flags.append(
             QualityFlag(
                 code="PDFPLUMBER_ERROR",
                 message="Critical error during table extraction.",
                 severity="error",
                 context={"error": str(exc)},
             )
         )

    return tables, flags
