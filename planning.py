from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from requests import HTTPError, RequestException

from ollama import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    queries: List[str]
    rationale: str = ""
    raw_plan: str = ""
    used_model: bool = False


class QueryPlanner:
    """LLM-backed search query planner."""

    def __init__(
        self,
        client: OllamaClient,
        model_name: str,
        max_queries: int = 5,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_queries = max(1, max_queries)
        self.max_retries = max(1, max_retries)

    def plan_queries(
        self,
        company: str,
        address: str,
        default_queries: List[str],
    ) -> QueryPlan:
        """Ask the model for a structured list of web-search queries."""

        prompt = self._build_prompt(company=company, address=address)
        plan = QueryPlan(queries=[], rationale="planner request failed")
        last_raw: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                raw = self._call_model(prompt, company, address)
                last_raw = raw
                candidate = self._parse_plan(raw)
                candidate.raw_plan = raw
                candidate.used_model = True

                if candidate.queries:
                    plan = candidate
                    break

                plan = candidate
            except RequestException as exc:
                logger.warning(
                    "Query planning request failed for %s | %s (attempt %d/%d): %s",
                    company,
                    address,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                plan = QueryPlan(queries=[], rationale="planner request failed")

            if attempt < self.max_retries - 1:
                logger.info(
                    "Planner retrying due to invalid JSON for %s | %s (attempt %d/%d)",
                    company,
                    address,
                    attempt + 1,
                    self.max_retries,
                )
                prompt = self._repair_prompt(company, address, last_raw or "")

        if not plan.queries:
            plan.queries = default_queries[: self.max_queries]
            if not plan.rationale:
                plan.rationale = "Fallback to heuristic queries"
            plan.used_model = plan.used_model and bool(plan.raw_plan)

        plan.queries = [q for q in plan.queries if q.strip()][: self.max_queries]
        if not plan.queries:
            plan.queries = default_queries[: self.max_queries]

        return plan

    def _call_model(self, prompt: str, company: str, address: str) -> str:
        base_options = {"temperature": 0.2, "num_ctx": 2048}
        json_options = {**base_options, "format": "json"}

        try:
            return self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=json_options,
            )
        except HTTPError as exc:
            logger.warning(
                "Query planner JSON mode failed for %s | %s: %s. Retrying without format.",
                company,
                address,
                exc,
            )
            return self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options=base_options,
            )

    def _repair_prompt(self, company: str, address: str, previous: str) -> str:
        base = self._build_prompt(company=company, address=address)
        return (
            f"{base}\nThe previous response was not valid JSON."
            " Re-read the instructions and respond with ONLY the JSON object."
            " Do not add commentary. Previous attempt was:\n"
            f"{previous}\n"
        )

    def _build_prompt(self, company: str, address: str) -> str:
        display_company = company or "Unknown company"
        display_address = address or "Unknown address"
        return (
            "You are a research planner helping classify industrial facilities.\n"
            "Given a company and address, produce targeted Exa web search"
            " queries that will reveal facility type, operations, and any"
            " regulatory information.\n\n"
            f"Company: {display_company}\n"
            f"Address: {display_address}\n\n"
            "Return STRICT JSON with this shape:\n"
            "{\n"
            "  \"queries\": [\"...\"],\n"
            "  \"rationale\": \"why these queries help\"\n"
            "}\n"
            "- Include at most {limit} focused queries.\n"
            "- Blend company, address, and facility keywords (e.g., manufacturing,"
            " distribution, headquarters).\n"
            "- Do not include commentary outside the JSON object."
        ).replace("{limit}", str(self.max_queries))

    def _parse_plan(self, raw: str) -> QueryPlan:
        cleaned = self._strip_markdown_fence(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Planner returned non-JSON output: %s", raw)
            return QueryPlan(queries=[], rationale="planner returned invalid JSON", raw_plan=raw)

        queries = []
        raw_queries = parsed.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(q).strip() for q in raw_queries if str(q).strip()]

        rationale = str(parsed.get("rationale", "")).strip()
        return QueryPlan(queries=queries, rationale=rationale, raw_plan=raw)

    def _strip_markdown_fence(self, raw: str) -> str:
        text = raw.strip()
        if not text.startswith("```"):
            return raw

        lines = text.splitlines()
        if not lines:
            return raw

        lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()

        return "\n".join(lines).strip() or raw
