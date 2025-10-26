import json
import logging
from typing import Dict, Any

from constants import ADDRESS_COLUMN, COMPANY_COLUMN
from ollama import OllamaClient

logger = logging.getLogger(__name__)


def build_prompt(company_name: str, address: str) -> str:
    """
    Placeholder prompt. Right now we're just exercising the pipe.

    Eventually this becomes:
    - "given this address and scraped evidence, classify site type"
    - and we'll enforce strict JSON contracts.

    For now we tell the model exactly what JSON to return so we can parse it.
    """
    display_company = company_name or "Unknown company"
    display_address = address or "Unknown address"
    return (
        "You are a helper for facility classification.\n"
        f"The company is:\n{display_company}\n"
        f"The site address is:\n{display_address}\n\n"
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

    logger.debug(
        "Building prompt for %s='%s', %s='%s'",
        COMPANY_COLUMN,
        company_name,
        ADDRESS_COLUMN,
        address,
    )
    prompt = build_prompt(company_name=company_name, address=address)

    raw_output = client.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    )
    logger.debug("Received raw model output for %s: %s", address, raw_output)

    # Try to parse strict JSON
    site_type = ""
    confidence = ""
    notes = ""

    try:
        parsed = json.loads(raw_output)
        site_type = str(parsed.get("site_type", ""))
        confidence = str(parsed.get("confidence", ""))
        notes = str(parsed.get("notes", ""))
    except json.JSONDecodeError:
        # We'll stash whatever came back for debugging
        logger.warning(
            "JSON decoding failed for address '%s'. Raw output: %s",
            address,
            raw_output,
        )
        notes = "model did not return valid JSON"

    return {
        "site_type": site_type,
        "confidence": confidence,
        "notes": notes,
        "raw_model_output": raw_output,
    }
