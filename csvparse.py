import csv
from pathlib import Path
from typing import List, Dict, Any

from classification import run_model_on_address
from ollama import OllamaClient


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
    extra_headers = ["site_type", "confidence", "notes", "raw_model_output"]

    fieldnames = original_headers[:]
    for h in extra_headers:
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
) -> None:
    """
    - Read input CSV
    - For each row:
        - call Ollama with placeholder prompt
        - merge new columns into the row
    - Write output CSV
    """

    # init client
    client = OllamaClient(base_url=ollama_url)

    # load input
    input_rows = read_input_csv(input_path)

    # detect headers so we preserve column order
    if input_rows:
        original_headers = list(input_rows[0].keys())
    else:
        original_headers = ["address"]

    output_rows: List[Dict[str, Any]] = []

    for row in input_rows:
        address = row.get("address", "").strip()
        model_result = run_model_on_address(
            address=address,
            client=client,
            model_name=model_name,
        )

        merged = {**row, **model_result}
        output_rows.append(merged)

    # save output
    write_output_csv(
        path=output_path,
        rows=output_rows,
        original_headers=original_headers,
    )
