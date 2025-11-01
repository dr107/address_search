from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from exa_py import Exa

from planning import QueryPlan, QueryPlanner

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TEXT_CHARS = 2_000
DEFAULT_FETCH_TEXT_CHARS = 12_000


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


@dataclass
class EvidenceDocument:
    url: str
    title: str
    snippet: str
    content: str


def _build_exa_client() -> Exa:
    token = os.environ.get("EXA_TOKEN")
    if not token:
        raise RuntimeError(
            "EXA_TOKEN environment variable is required to use the Exa SDK."
        )
    return Exa(api_key=token)


class ExaSearchClient:
    """
    Lightweight wrapper around the Exa SDK for search + optional content snippets.
    """

    def __init__(
        self,
        *,
        exa: Optional[Exa] = None,
        search_text_chars: int = DEFAULT_SEARCH_TEXT_CHARS,
    ):
        self._exa = exa or _build_exa_client()
        self._search_text_chars = max(256, search_text_chars)

    @property
    def exa(self) -> Exa:
        return self._exa

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query:
            return []

        limit = max(1, min(20, max_results))
        logger.debug("Searching Exa for query=%s (limit=%d)", query, limit)

        try:
            response = self._exa.search(
                query=query,
                num_results=limit,
                contents={"text": {"max_characters": self._search_text_chars}},
                type="neural",
            )
        except Exception as exc:
            logger.warning("Exa search failed for '%s': %s", query, exc)
            return []

        items = response.results or []
        results: List[SearchResult] = []
        for item in items:
            url = getattr(item, "url", "") or ""
            if not url:
                continue
            title = (getattr(item, "title", None) or "").strip()
            snippet_source = (
                getattr(item, "summary", None)
                or getattr(item, "text", None)
                or getattr(item, "extras", {})
            )
            if isinstance(snippet_source, dict):
                snippet = str(snippet_source.get("summary") or "").strip()
            else:
                snippet = str(snippet_source or "").strip()
            if not snippet and getattr(item, "text", None):
                snippet = str(item.text).strip()
            if snippet:
                snippet = snippet[:1000]

            results.append(SearchResult(url=url, title=title, snippet=snippet))
            if len(results) >= limit:
                break

        logger.debug("Exa returned %d result(s) for query=%s", len(results), query)
        return results

    def fetch_remote_content(
        self,
        url: str,
        *,
        max_characters: int = DEFAULT_FETCH_TEXT_CHARS,
    ) -> Optional[str]:
        if not url:
            return None

        try:
            response = self._exa.get_contents(
                urls=[url],
                text={"max_characters": max(512, max_characters)},
            )
        except Exception as exc:
            logger.warning("Exa get_contents failed for %s: %s", url, exc)
            return None

        for result in response.results or []:
            text = getattr(result, "text", None)
            if text:
                return str(text)
        return None


class ExaContentFetcher:
    """
    Fetches page content via the Exa SDK.
    """

    def __init__(
        self,
        *,
        exa: Optional[Exa] = None,
        max_characters: int = DEFAULT_FETCH_TEXT_CHARS,
    ):
        self._exa = exa or _build_exa_client()
        self._max_characters = max(512, max_characters)

    def fetch(self, url: str) -> str:
        if not url:
            raise ValueError("URL is required for Exa content fetch.")

        try:
            response = self._exa.get_contents(
                urls=[url],
                text={"max_characters": self._max_characters},
            )
        except Exception as exc:
            raise RuntimeError(f"Exa get_contents failed for {url}: {exc}") from exc

        for result in response.results or []:
            text = getattr(result, "text", None)
            if text:
                return str(text)

        raise RuntimeError(f"Exa returned no content for {url}")


class ResearchPipeline:
    """
    Tie search + fetch together to gather evidence snippets.
    """

    def __init__(
        self,
        search_client: ExaSearchClient,
        page_fetcher: ExaContentFetcher,
        max_search_results: int = 5,
        max_documents: int = 3,
        planner: Optional[QueryPlanner] = None,
        expanded_queries: bool = False,
    ):
        self.search_client = search_client
        self.page_fetcher = page_fetcher
        self.max_search_results = max(1, max_search_results)
        self.max_documents = max(1, max_documents)
        self.planner = planner
        self.max_workers = max(1, self.max_documents * 2)
        self.expanded_queries = expanded_queries

    def build_queries(self, company: str, address: str) -> List[str]:
        company = (company or "").strip()
        address = (address or "").strip()
        queries: List[str] = []

        if company and address:
            queries.append(f'"{company}" "{address}"')
            if self.expanded_queries:
                first_segment = address.split(",")[0].strip()
                if first_segment:
                    queries.append(f'"{company}" "{first_segment}" facility')
                queries.append(f'"{company}" facility types')
        elif company:
            queries.append(f'"{company}"')
            if self.expanded_queries:
                queries.append(f'"{company}" facility types')
        elif address:
            queries.append(f'"{address}"')
            if self.expanded_queries:
                first_segment = address.split(",")[0].strip()
                if first_segment:
                    queries.append(f'"{first_segment}" facility')

        deduped: List[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in deduped:
                deduped.append(q)
        logger.debug(
            "Queries for company=%s address=%s (expanded=%s) -> %s",
            company,
            address,
            self.expanded_queries,
            deduped,
        )
        return deduped

    def collect_evidence(
        self, company: str, address: str
    ) -> Tuple[List[EvidenceDocument], QueryPlan, int, int]:
        if not company and not address:
            logger.debug("No company/address supplied for research; skipping.")
            empty_plan = QueryPlan(
                queries=[], rationale="Missing company and address", used_model=False
            )
            return [], empty_plan, 0, 0

        documents: List[EvidenceDocument] = []
        seen_urls: set[str] = set()
        search_call_count = 0
        fetch_call_count = 0

        base_queries = self.build_queries(company, address)
        if self.planner:
            plan = self.planner.plan_queries(
                company=company,
                address=address,
                default_queries=base_queries,
            )
            queries = plan.queries or base_queries
        else:
            plan = QueryPlan(
                queries=base_queries,
                rationale="Heuristic query builder",
                used_model=False,
            )
            queries = base_queries

        logger.debug(
            "Query plan for %s | %s -> %s",
            company,
            address,
            queries,
        )

        aggregated_results: List[SearchResult] = []
        search_workers = min(len(queries), self.max_workers)

        if search_workers > 1:
            with ThreadPoolExecutor(max_workers=search_workers) as executor:
                future_map = {
                    executor.submit(self._safe_search, query): query for query in queries
                }
                for future in as_completed(future_map):
                    results, calls = future.result()
                    aggregated_results.extend(results)
                    search_call_count += calls
        else:
            for query in queries:
                results, calls = self._safe_search(query)
                aggregated_results.extend(results)
                search_call_count += calls

        fetch_candidates: List[SearchResult] = []
        for result in aggregated_results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            fetch_candidates.append(result)

        if not fetch_candidates:
            logger.debug(
                "Collected %d evidence document(s) for company=%s address=%s",
                len(documents),
                company,
                address,
            )
            return documents, plan, search_call_count, fetch_call_count

        fetch_workers = min(self.max_workers, max(1, self.max_documents * 2))

        if fetch_workers > 1:
            with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
                futures = [
                    executor.submit(self._fetch_document, item) for item in fetch_candidates
                ]
                for future in as_completed(futures):
                    if len(documents) >= self.max_documents:
                        break
                    doc, calls = future.result()
                    fetch_call_count += calls
                    if doc:
                        documents.append(doc)
        else:
            for result in fetch_candidates:
                doc, calls = self._fetch_document(result)
                fetch_call_count += calls
                if doc:
                    documents.append(doc)
                if len(documents) >= self.max_documents:
                    break

        logger.debug(
            "Collected %d evidence document(s) for company=%s address=%s",
            len(documents),
            company,
            address,
        )
        return documents, plan, search_call_count, fetch_call_count

    def _safe_search(self, query: str) -> Tuple[List[SearchResult], int]:
        if not query:
            return [], 0
        try:
            results = self.search_client.search(
                query=query, max_results=self.max_search_results
            )
            return results, 1
        except Exception as exc:
            logger.warning("Search request failed for '%s': %s", query, exc)
            return [], 1

    def _fetch_document(self, result: SearchResult) -> Tuple[Optional[EvidenceDocument], int]:
        fetch_calls = 0
        try:
            fetch_calls += 1
            content = self.page_fetcher.fetch(result.url)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", result.url, exc)
            content = None
            if self.search_client:
                try:
                    fetch_calls += 1
                    content = self.search_client.fetch_remote_content(result.url)
                except Exception as fallback_exc:
                    logger.warning("fetch_content fallback failed for %s: %s", result.url, fallback_exc)
                    content = None

        if not content:
            return None, fetch_calls

        return EvidenceDocument(
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            content=content,
        ), fetch_calls
