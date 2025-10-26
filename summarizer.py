from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from requests import RequestException

from ollama import OllamaClient
from research import EvidenceDocument

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSummary:
    text: str
    source_count: int
    used_model: bool
    raw_output: str = ""


class EvidenceSummarizer:
    """Generate concise summaries of collected evidence snippets."""

    def __init__(
        self,
        client: OllamaClient,
        model_name: str,
        max_documents: int = 3,
        max_chars_per_doc: int = 600,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_documents = max(1, max_documents)
        self.max_chars_per_doc = max(200, max_chars_per_doc)

    def summarize(
        self,
        company: str,
        address: str,
        evidence: List[EvidenceDocument],
    ) -> EvidenceSummary:
        if not evidence:
            return EvidenceSummary(
                text="No supporting evidence collected yet.",
                source_count=0,
                used_model=False,
            )

        condensed_blocks = []
        for idx, doc in enumerate(evidence[: self.max_documents], start=1):
            snippet = doc.content[: self.max_chars_per_doc]
            condensed_blocks.append(
                f"Document {idx}: {doc.title}\nURL: {doc.url}\nSnippet: {doc.snippet}\nContent: {snippet}"
            )

        prompt = self._build_prompt(company, address, condensed_blocks)
        try:
            raw = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.15, "num_ctx": 4096},
            )
            text = raw.strip()
            used_model = True
        except RequestException as exc:
            logger.warning("Evidence summarizer failed for %s | %s: %s", company, address, exc)
            text = self._fallback_summary(evidence)
            raw = text
            used_model = False

        return EvidenceSummary(
            text=text,
            source_count=min(len(evidence), self.max_documents),
            used_model=used_model,
            raw_output=raw,
        )

    def _build_prompt(self, company: str, address: str, documents: List[str]) -> str:
        display_company = company or "Unknown company"
        display_address = address or "Unknown address"
        docs_blob = "\n\n".join(documents)
        return (
            "You are assisting an industrial-site classification pipeline."
            " Summarize the key operational clues from the following research"
            " documents in 3 bullet points or less. Each bullet should cite the"
            " document number when referencing facts. Keep it under 120 words.\n\n"
            f"Company: {display_company}\n"
            f"Address: {display_address}\n\n"
            "Research Documents:\n"
            f"{docs_blob}\n\n"
            "Return plain text bullet(s); do not emit JSON."
        )

    def _fallback_summary(self, evidence: List[EvidenceDocument]) -> str:
        snippets = []
        for idx, doc in enumerate(evidence[: self.max_documents], start=1):
            snippets.append(f"{idx}. {doc.title} — {doc.snippet}")
        return "; ".join(snippets) or "Evidence available but summarizer failed."
