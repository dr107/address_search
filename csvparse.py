import csv
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from agentic import AgenticConfig
from classification import run_model_on_address
from constants import (
    ADDRESS_COLUMN,
    COMPANY_COLUMN,
    DEFAULT_AGENTIC_MODEL_HINTS,
    OUTPUT_EXTRA_HEADERS,
)
from ollama import OllamaClient
from planning import QueryPlan, QueryPlanner
from research import EvidenceDocument, ExaContentFetcher, ExaSearchClient, ResearchPipeline
from summarizer import EvidenceSummarizer

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


def _persist_progress(
    output_rows: List[Dict[str, Any]],
    original_headers: List[str],
    output_path: Path,
) -> None:
    if not output_rows:
        return
    logger.info(
        "Persisting %d row(s) to %s", len(output_rows), output_path
    )
    write_output_csv(
        path=output_path,
        rows=output_rows,
        original_headers=original_headers,
    )


def _cache_key(company: str, address: str) -> Optional[Tuple[str, str]]:
    company_key = (company or "").strip().lower()
    address_key = (address or "").strip().lower()
    if not company_key and not address_key:
        return None
    return (company_key, address_key)


def load_cached_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}

    logger.info("Loading cached results from %s", path)
    cached_data = read_input_csv(path)
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in cached_data:
        key = _cache_key(row.get(COMPANY_COLUMN, ""), row.get(ADDRESS_COLUMN, ""))
        if key:
            cache[key] = row
    logger.info("Loaded %d cached row(s).", len(cache))
    return cache


def format_query_plan(plan: Optional[QueryPlan]) -> str:
    if not plan:
        return ""

    parts = []
    rationale = plan.rationale.strip()
    if rationale:
        parts.append(rationale)

    if plan.queries:
        parts.append("Queries: " + "; ".join(plan.queries))

    return " | ".join(parts)


def format_category_signature(categories: Optional[List[Dict[str, str]]]) -> str:
    if not categories:
        return ""

    parts = []
    for entry in categories:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        description = str(entry.get("description", "") or "").strip()
        if description:
            parts.append(f"{name}: {description}")
        else:
            parts.append(name)
    return " | ".join(parts)


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


def _needs_expanded_search(result: Dict[str, Any]) -> bool:
    site_type = str(result.get("site_type", "") or "").strip().lower()
    if not site_type or site_type == "unknown":
        return True
    notes = str(result.get("notes", "") or "").strip().lower()
    if any(phrase in notes for phrase in ["insufficient", "not enough", "unable to determine", "no evidence"]):
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
    agentic_mode: str = "auto",
    agentic_model_hints: Optional[List[str]] = None,
    agent_max_iterations: int = 6,
    ignore_cache: bool = False,
    category_suggestions: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 5,
    random_sample: bool = False,
    expanded_search_results: Optional[int] = None,
    expanded_max_documents: Optional[int] = None,
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

    query_planner: Optional[QueryPlanner] = None
    evidence_summarizer: Optional[EvidenceSummarizer] = None
    if enable_web_research:
        query_planner = QueryPlanner(client=client, model_name=model_name)
        evidence_summarizer = EvidenceSummarizer(client=client, model_name=model_name)

    category_hint_list = category_suggestions or []
    category_hint_text = format_category_signature(category_hint_list)

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

    search_client: Optional[ExaSearchClient] = None
    fetcher: Optional[ExaContentFetcher] = None
    research_pipeline: Optional[ResearchPipeline] = None

    needs_search_stack = enable_web_research or use_agentic_tools
    if needs_search_stack:
        logger.info("Initializing Exa search stack.")
        search_client = ExaSearchClient()
        fetcher = ExaContentFetcher(exa=search_client.exa)

    if enable_web_research and search_client and fetcher:
        if use_agentic_tools:
            logger.info(
                "Web research pipeline enabled alongside agentic tools (max_search_results=%d, max_documents=%d)",
                max_search_results,
                max_documents,
            )
        else:
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
            planner=None,
            expanded_queries=False,
        )
        if query_planner:
            logger.debug(
                "Initial research pass uses heuristic combined queries; planner reserved for expanded retries."
            )
    elif not enable_web_research:
        logger.info("Web research pipeline disabled for this run.")
    else:
        logger.info("Web research pipeline unavailable: search stack not initialized.")

    logger.info("Loading input CSV from %s", input_path)
    input_rows = read_input_csv(input_path)
    logger.info("Loaded %d rows from %s", len(input_rows), input_path)

    cached_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not ignore_cache:
        cached_rows = load_cached_rows(output_path)
    else:
        logger.info("Cache disabled via --ignore-cache")

    # detect headers so we preserve column order
    if input_rows:
        original_headers = list(input_rows[0].keys())
    else:
        original_headers = [ADDRESS_COLUMN]

    output_rows: List[Dict[str, Any]] = []
    rows_since_flush = 0
    batch_size = max(1, batch_size)

    if limit is not None:
        if limit < 0:
            logger.warning("Limit %d is negative; treating as 0.", limit)
            limit = 0
        effective_limit = min(limit, len(input_rows))
        if random_sample and effective_limit > 0:
            logger.info(
                "Processing %d randomly sampled row(s) out of %d.",
                effective_limit,
                len(input_rows),
            )
            rows_to_process = random.sample(input_rows, k=effective_limit)
        else:
            logger.info(
                "Processing the first %d row(s).",
                effective_limit,
            )
            rows_to_process = input_rows[:effective_limit]
    else:
        rows_to_process = input_rows

    logger.info("Beginning processing for %d row(s).", len(rows_to_process))

    total_search_calls = 0
    total_fetch_calls = 0
    log_interval = max(1, batch_size)

    for idx, row in enumerate(rows_to_process, start=1):
        address = row.get(ADDRESS_COLUMN, "").strip()
        company = row.get(COMPANY_COLUMN, "").strip()
        if not address:
            logger.warning(
                "Row %d missing '%s' field; skipping model call.", idx, ADDRESS_COLUMN
            )

        cache_key = _cache_key(company, address)
        cache_hit = cached_rows.get(cache_key) if cache_key else None

        if cache_hit and cache_hit.get("category_suggestions", "") == category_hint_text:
            logger.debug(
                "Cache hit for row %d (%s | %s); reusing previous classification.",
                idx,
                company,
                address,
            )
            merged = {**row}
            for header in OUTPUT_EXTRA_HEADERS:
                if header in cache_hit:
                    merged[header] = cache_hit[header]
            output_rows.append(merged)
            rows_since_flush += 1
            if rows_since_flush >= batch_size:
                _persist_progress(
                    output_rows=output_rows,
                    original_headers=original_headers,
                    output_path=output_path,
                )
                rows_since_flush = 0
            continue
        elif cache_hit:
            logger.debug(
                "Cache miss for row %d due to category change (%s | %s).",
                idx,
                company,
                address,
            )

        evidence_docs: List[EvidenceDocument] = []
        query_plan: Optional[QueryPlan] = None
        if research_pipeline:
            evidence_docs, query_plan, search_calls, fetch_calls = research_pipeline.collect_evidence(
                company=company, address=address
            )
            total_search_calls += search_calls
            total_fetch_calls += fetch_calls

        original_plan_summary = format_query_plan(query_plan)
        plan_summary = original_plan_summary

        evidence_summary_text = ""
        if evidence_summarizer and evidence_docs:
            summary = evidence_summarizer.summarize(
                company=company,
                address=address,
                evidence=evidence_docs,
            )
            evidence_summary_text = summary.text

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
            category_suggestions=category_hint_list if category_hint_list else None,
            agent_config=agent_config,
            evidence_summary=evidence_summary_text,
        )

        can_expand = (
            _needs_expanded_search(model_result)
            and search_client is not None
            and fetcher is not None
        )

        if can_expand:
            expanded_results = (
                expanded_search_results
                if expanded_search_results is not None
                else max_search_results * 2
            )
            expanded_results = max(expanded_results, max_search_results)
            expanded_docs_limit = (
                expanded_max_documents
                if expanded_max_documents is not None
                else max_documents * 2
            )
            expanded_docs_limit = max(expanded_docs_limit, max_documents)

            current_max_results = (
                research_pipeline.max_search_results if research_pipeline else max_search_results
            )
            current_max_documents = (
                research_pipeline.max_documents if research_pipeline else max_documents
            )
            current_expanded = research_pipeline.expanded_queries if research_pipeline else False

            should_expand = (
                not current_expanded
                or expanded_results > current_max_results
                or expanded_docs_limit > current_max_documents
                or not evidence_docs
            )

            if should_expand:
                logger.info(
                    "Row %d: classification inconclusive; expanding search (max_search_results=%d, max_documents=%d).",
                    idx,
                    expanded_results,
                    expanded_docs_limit,
                )
                expanded_pipeline = ResearchPipeline(
                    search_client=search_client,
                    page_fetcher=fetcher,
                    max_search_results=expanded_results,
                    max_documents=expanded_docs_limit,
                    planner=query_planner,
                    expanded_queries=True,
                )
                new_evidence_docs, new_plan, new_search_calls, new_fetch_calls = expanded_pipeline.collect_evidence(
                    company=company, address=address
                )
                total_search_calls += new_search_calls
                total_fetch_calls += new_fetch_calls
                if new_evidence_docs:
                    evidence_docs = new_evidence_docs
                    new_plan_summary = format_query_plan(new_plan)
                    if plan_summary and new_plan_summary and new_plan_summary != plan_summary:
                        plan_summary = f"{plan_summary} || Expanded: {new_plan_summary}"
                    elif new_plan_summary:
                        plan_summary = f"Expanded: {new_plan_summary}"

                    if evidence_summarizer:
                        summary = evidence_summarizer.summarize(
                            company=company,
                            address=address,
                            evidence=evidence_docs,
                        )
                        evidence_summary_text = summary.text

                    model_result = run_model_on_address(
                        address=address,
                        company_name=company,
                        client=client,
                        model_name=model_name,
                        evidence=evidence_docs,
                        category_suggestions=category_hint_list if category_hint_list else None,
                        agent_config=agent_config,
                        evidence_summary=evidence_summary_text,
                    )
                else:
                    logger.info(
                        "Row %d: expanded search produced no additional evidence.",
                        idx,
                    )
            else:
                logger.debug(
                    "Row %d: expanded search thresholds not higher than base; skipping expansion.",
                    idx,
                )

        merged = {**row, **model_result}
        if plan_summary:
            merged["query_plan"] = plan_summary
        if evidence_summary_text:
            merged["evidence_summary"] = evidence_summary_text
        if category_hint_text:
            merged["category_suggestions"] = category_hint_text
        elif "category_suggestions" not in merged:
            merged["category_suggestions"] = ""
        output_rows.append(merged)
        rows_since_flush += 1

        if idx % log_interval == 0:
            logger.debug(
                "Processed %d row(s); cumulative Exa usage: %d searches, %d fetches.",
                idx,
                total_search_calls,
                total_fetch_calls,
            )

        if rows_since_flush >= batch_size:
            _persist_progress(
                output_rows=output_rows,
                original_headers=original_headers,
                output_path=output_path,
            )
            rows_since_flush = 0

    if output_rows:
        if rows_since_flush > 0:
            _persist_progress(
                output_rows=output_rows,
                original_headers=original_headers,
                output_path=output_path,
            )
    else:
        logger.info("No rows processed; writing empty output file to %s", output_path)
        write_output_csv(
            path=output_path,
            rows=[],
            original_headers=original_headers,
        )

    logger.info(
        "Finished writing %s (total Exa usage: %d searches, %d fetches)",
        output_path,
        total_search_calls,
        total_fetch_calls,
    )
