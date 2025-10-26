import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from classification import run_model_on_address
from constants import ADDRESS_COLUMN, COMPANY_COLUMN, OUTPUT_EXTRA_HEADERS
from ollama import OllamaClient

logger = logging.getLogger(__name__)


def read_input_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read CSV into a list[dict]. We expect an 'address' column at minimum.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def write_output_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    original_headers: List[str],
) -> None:
    """
    Write enriched CSV. We keep original columns first,
    then add our new columns if they don't already exist.
    """
    fieldnames = original_headers[:]
    for h in OUTPUT_EXTRA_HEADERS:
        if h not in fieldnames:
            fieldnames.append(h)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_file(
    input_path: Path,
    output_path: Path,
    ollama_url: str,
    model_name: str,
    limit: Optional[int] = None,
) -> None:
    """
    - Read input CSV
    - For each row:
        - call Ollama with placeholder prompt
        - merge new columns into the row
    - Write output CSV
    """

    logger.info("Initializing Ollama client (%s)", ollama_url)
    client = OllamaClient(base_url=ollama_url)

    logger.info("Loading input CSV from %s", input_path)
    input_rows = read_input_csv(input_path)
    logger.info("Loaded %d rows from %s", len(input_rows), input_path)

    # detect headers so we preserve column order
    if input_rows:
        original_headers = list(input_rows[0].keys())
    else:
        original_headers = [ADDRESS_COLUMN]

    output_rows: List[Dict[str, Any]] = []

    if limit is not None:
        if limit < 0:
            logger.warning("Limit %d is negative; treating as 0.", limit)
            limit = 0
        logger.info("Processing at most %d rows.", limit)
        rows_to_process = input_rows[:limit]
    else:
        rows_to_process = input_rows

    logger.info("Beginning processing for %d row(s).", len(rows_to_process))

    for idx, row in enumerate(rows_to_process, start=1):
        address = row.get(ADDRESS_COLUMN, "").strip()
        company = row.get(COMPANY_COLUMN, "").strip()
        if not address:
            logger.warning(
                "Row %d missing '%s' field; skipping model call.", idx, ADDRESS_COLUMN
            )

        model_result = run_model_on_address(
            address=address,
            company_name=company,
            client=client,
            model_name=model_name,
        )

        merged = {**row, **model_result}
        output_rows.append(merged)

    logger.info("Writing %d enriched row(s) to %s", len(output_rows), output_path)
    write_output_csv(
        path=output_path,
        rows=output_rows,
        original_headers=original_headers,
    )
    logger.info("Finished writing %s", output_path)
