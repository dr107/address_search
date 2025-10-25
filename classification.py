import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from ollama import OllamaClient

def build_prompt_for_address(address: str) -> str:
    """
    Placeholder prompt. Right now we're just exercising the pipe.

    Eventually this becomes:
    - "given this address and scraped evidence, classify site type"
    - and we'll enforce strict JSON contracts.

    For now we tell the model exactly what JSON to return so we can parse it.
    """
    return (
        "You are a helper for facility classification.\n"
        f"The site address is:\n{address}\n\n"
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
    client: OllamaClient,
    model_name: str,
) -> Dict[str, Any]:
    """
    Ask the model about this address (placeholder behavior)
    and parse the JSON it returns.

    We try to be robust if the model doesn't obey perfectly.
    """

    if not address:
        return {
            "site_type": "",
            "confidence": "",
            "notes": "no address provided",
            "raw_model_output": "",
        }

    prompt = build_prompt_for_address(address)
    raw_output = client.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    )

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
        notes = "model did not return valid JSON"

    return {
        "site_type": site_type,
        "confidence": confidence,
        "notes": notes,
        "raw_model_output": raw_output,
    }
