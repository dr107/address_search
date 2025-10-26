import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentic import AgenticConfig
from classification import run_model_on_address
from constants import (
    ADDRESS_COLUMN,
    COMPANY_COLUMN,
    DEFAULT_AGENTIC_MODEL_HINTS,
    OUTPUT_EXTRA_HEADERS,
)
from ollama import OllamaClient
from research import DuckDuckGoAPIClient, PageFetcher, ResearchPipeline

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


def _should_use_agentic_loop(
    agentic_mode: str,
    model_name: str,
    hints: List[str],
) -> bool:
    normalized = model_name.lower()
    if agentic_mode == "on":
        return True
    if agentic_mode == "off":
        return False

    for hint in hints:
        if hint and hint.lower() in normalized:
            return True
    return False


def process_file(
    input_path: Path,
    output_path: Path,
    ollama_url: str,
    model_name: str,
    limit: Optional[int] = None,
    enable_web_research: bool = False,
    max_search_results: int = 5,
    max_documents: int = 3,
    fetch_timeout: int = 20,
    search_api_url: str = "http://localhost:8000",
    agentic_mode: str = "auto",
    agentic_model_hints: Optional[List[str]] = None,
    agent_max_iterations: int = 6,
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

    hints = agentic_model_hints or DEFAULT_AGENTIC_MODEL_HINTS
    use_agentic_tools = _should_use_agentic_loop(
        agentic_mode=agentic_mode,
        model_name=model_name,
        hints=hints,
    )
    if agentic_mode == "auto":
        logger.info(
            "Agentic mode auto-detected as %s based on model '%s'.",
            "ENABLED" if use_agentic_tools else "DISABLED",
            model_name,
        )
    else:
        logger.info("Agentic mode explicitly set to %s.", agentic_mode.upper())

    search_client: Optional[DuckDuckGoAPIClient] = None
    fetcher: Optional[PageFetcher] = None
    research_pipeline: Optional[ResearchPipeline] = None

    needs_search_stack = enable_web_research or use_agentic_tools
    if needs_search_stack:
        logger.info("Initializing search stack via %s", search_api_url)
        search_client = DuckDuckGoAPIClient(
            base_url=search_api_url,
            timeout=fetch_timeout,
        )
        fetcher = PageFetcher(timeout=fetch_timeout)

    if enable_web_research and not use_agentic_tools and search_client and fetcher:
        logger.info(
            "Web research (pipeline) enabled (max_search_results=%d, max_documents=%d)",
            max_search_results,
            max_documents,
        )
        research_pipeline = ResearchPipeline(
            search_client=search_client,
            page_fetcher=fetcher,
            max_search_results=max_search_results,
            max_documents=max_documents,
        )
    elif not enable_web_research:
        logger.info("Web research pipeline disabled for this run.")
    else:
        logger.info("Web research pipeline skipped because agentic tools are active.")

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

        evidence_docs = []
        if research_pipeline:
            evidence_docs = research_pipeline.collect_evidence(
                company=company, address=address
            )

        agent_config = AgenticConfig(
            enabled=use_agentic_tools,
            search_client=search_client,
            page_fetcher=fetcher,
            max_iterations=agent_max_iterations,
        )

        model_result = run_model_on_address(
            address=address,
            company_name=company,
            client=client,
            model_name=model_name,
            evidence=evidence_docs,
            agent_config=agent_config,
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
