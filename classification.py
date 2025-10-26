import json
import logging
from typing import Any, Dict, List, Optional

from agentic import AgenticConfig, ToolAgent, ToolAgentError
from constants import ADDRESS_COLUMN, COMPANY_COLUMN
from ollama import OllamaClient
from research import EvidenceDocument
from requests import HTTPError

logger = logging.getLogger(__name__)


def _format_evidence(evidence: Optional[List[EvidenceDocument]]) -> str:
    if not evidence:
        return "No supporting evidence was gathered."

    chunks = []
    for idx, doc in enumerate(evidence, start=1):
        content_preview = doc.content[:800]
        chunks.append(
            f"{idx}. Title: {doc.title}\n"
            f"   URL: {doc.url}\n"
            f"   Snippet: {doc.snippet}\n"
            f"   Content Preview: {content_preview}"
        )
    return "\n".join(chunks)


def build_prompt(
    company_name: str,
    address: str,
    evidence: Optional[List[EvidenceDocument]],
    summary: Optional[str] = None,
    category_suggestions: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Placeholder prompt enriched with gathered evidence.
    """
    display_company = company_name or "Unknown company"
    display_address = address or "Unknown address"
    evidence_text = _format_evidence(evidence)
    summary_text = summary.strip() if summary else "No summarized evidence was available."
    if category_suggestions:
        guidance_lines = []
        for entry in category_suggestions:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            description = str(entry.get("description", "") or "").strip()
            if description:
                guidance_lines.append(f"- {name}: {description}")
            else:
                guidance_lines.append(f"- {name}")
        guidance_blob = "\n".join(guidance_lines)
        category_text = (
            "Use one of the suggested site_type categories below when possible."
            " If none fit, you may craft a new category that better matches the evidence.\n"
            f"Suggested categories:\n{guidance_blob}"
        )
    else:
        category_text = (
            "You may define whatever site_type category best matches the facility."
            " Ensure the label is concise and descriptive."
        )
    return (
        "You are a helper for facility classification.\n"
        "Using the research summary and evidence below, determine what kind of site this is.\n"
        f"Company: {display_company}\n"
        f"Address: {display_address}\n\n"
        f"Category guidance:\n{category_text}\n\n"
        "Research summary:\n"
        f"{summary_text}\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        "Return ONLY strict JSON with these keys:\n"
        '{\n'
        '  "site_type": "unknown",\n'
        '  "confidence": "low",\n'
        '  "notes": "placeholder - pipeline not implemented yet"\n'
        '}\n'
        "Do not include any extra commentary.\n"
    )


def run_model_on_address(
    address: str,
    company_name: str,
    client: OllamaClient,
    model_name: str,
    evidence: Optional[List[EvidenceDocument]] = None,
    category_suggestions: Optional[List[Dict[str, str]]] = None,
    evidence_summary: Optional[str] = None,
    agent_config: Optional[AgenticConfig] = None,
) -> Dict[str, Any]:
    """
    Ask the model about this address (placeholder behavior)
    and parse the JSON it returns.

    We try to be robust if the model doesn't obey perfectly.
    """

    if not address:
        logger.debug(
            "No '%s' provided; returning empty stub result.", ADDRESS_COLUMN
        )
        return {
            "site_type": "",
            "confidence": "",
            "notes": f"no {ADDRESS_COLUMN} provided",
            "raw_model_output": "",
        }

    use_agentic = (
        agent_config
        and agent_config.enabled
        and agent_config.search_client
        and agent_config.page_fetcher
    )

    context = f"{company_name} | {address}"

    if use_agentic:
        logger.info(
            "Running agentic classification for %s (%s)", company_name, address
        )
        tool_agent = ToolAgent(
            client=client,
            model_name=model_name,
            search_client=agent_config.search_client,
            page_fetcher=agent_config.page_fetcher,
            max_iterations=agent_config.max_iterations,
        )
        try:
            raw_output = tool_agent.classify(
                company_name=company_name,
                address=address,
                categories=category_suggestions,
            )
            return _parse_model_response(raw_output, context=context)
        except ToolAgentError as exc:
            logger.warning(
                "Agentic mode failed for %s (%s): %s. Falling back to single-shot prompt.",
                company_name,
                address,
                exc,
            )

    logger.debug(
        "Building prompt for %s='%s', %s='%s'",
        COMPANY_COLUMN,
        company_name,
        ADDRESS_COLUMN,
        address,
    )
    prompt = build_prompt(
        company_name=company_name,
        address=address,
        evidence=evidence,
        summary=evidence_summary,
        category_suggestions=category_suggestions,
    )

    base_options = {
        "temperature": 0.1,
        "num_ctx": 4096,
    }
    json_options = {**base_options, "format": "json"}

    try:
        raw_output = client.generate(
            model=model_name,
            prompt=prompt,
            options=json_options,
        )
    except HTTPError as exc:
        logger.warning(
            "JSON-enforced generation failed for %s (%s); retrying without format constraint: %s",
            company_name,
            address,
            exc,
        )
        raw_output = client.generate(
            model=model_name,
            prompt=prompt,
            options=base_options,
        )
    logger.debug("Received raw model output for %s: %s", address, raw_output)

    context = f"{company_name} | {address}"
    return _parse_model_response(raw_output, context=context)


def _parse_model_response(raw_output: str, context: str = "") -> Dict[str, Any]:
    # Try to parse strict JSON
    site_type = ""
    confidence = ""
    notes = ""

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = _extract_json_blob(raw_output)
        if parsed is None:
            if context:
                logger.warning(
                    "JSON decoding failed for %s. Raw output: %s", context, raw_output
                )
            else:
                logger.warning("JSON decoding failed. Raw output: %s", raw_output)
            notes = "model did not return valid JSON"
        else:
            logger.debug("Recovered JSON blob from noisy output for %s", context)

    if parsed:
        site_type = str(parsed.get("site_type", ""))
        confidence = str(parsed.get("confidence", ""))
        notes = str(parsed.get("notes", notes))

    return {
        "site_type": site_type,
        "confidence": confidence,
        "notes": notes,
        "raw_model_output": raw_output,
    }


def _extract_json_blob(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to recover the first JSON object embedded in a noisy string.
    """
    stack = 0
    start_idx: Optional[int] = None
    for idx, ch in enumerate(raw_output):
        if ch == "{":
            if stack == 0:
                start_idx = idx
            stack += 1
        elif ch == "}":
            if stack:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    snippet = raw_output[start_idx : idx + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        continue
    return None
