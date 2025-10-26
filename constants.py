"""Centralized configuration for column names and output headers."""

ADDRESS_COLUMN = "Full Address"
COMPANY_COLUMN = "Company Name"

# Extra columns appended to the enriched CSV output.
OUTPUT_EXTRA_HEADERS = [
    "site_type",
    "confidence",
    "notes",
    "raw_model_output",
    "query_plan",
    "evidence_summary",
]

# Models we assume are capable of agentic/tool use when --agentic-mode=auto.
DEFAULT_AGENTIC_MODEL_HINTS = [
    "llama3",
    "llama-3",
    "llama4",
    "llama-4",
    "deepseek",
]
