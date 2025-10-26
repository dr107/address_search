from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from planning import QueryPlan, QueryPlanner

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


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


class DuckDuckGoAPIClient:
    """
    Client for the local DuckDuckGo MCP server exposed via OpenAPI.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 15,
        safe_search: str = "moderate",
    ):
        self.headers = {"User-Agent": USER_AGENT}
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.safe_search = safe_search

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        logger.debug(
            "Searching DuckDuckGo API for query=%s (count=%d)", query, max_results
        )
        payload = {
            "query": query,
            "max_results": max(1, min(20, max_results)),
        }
        resp = requests.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=self.timeout,
            headers=self.headers,
        )
        resp.raise_for_status()

        data = resp.json()
        items: Optional[List[Any]] = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                items = data["results"]
            elif "data" in data and isinstance(data["data"], list):
                items = data["data"]

        results: List[SearchResult] = []

        if not items:
            blob: Optional[str] = None
            if isinstance(data, str):
                blob = data
            elif (
                isinstance(data, dict)
                and isinstance(data.get("result"), str)
            ):
                blob = data["result"]

            if blob:
                items = self._parse_structured_text(blob, max_results)

        if not items:
            logger.debug("DuckDuckGo API returned no parsable results for %s", query)
            return results

        for item in items:
            if isinstance(item, SearchResult):
                results.append(item)
                if len(results) >= max_results:
                    break
                continue

            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "") or item.get("name", "")).strip()
            snippet = str(
                item.get("snippet", "")
                or item.get("description", "")
                or item.get("excerpt", "")
            ).strip()
            if not url:
                continue
            results.append(SearchResult(url=url, title=title, snippet=snippet))
            if len(results) >= max_results:
                break

        logger.debug(
            "DuckDuckGo API returned %d result(s) for %s", len(results), query
        )
        return results

    @staticmethod
    def _parse_structured_text(blob: str, limit: int) -> List[SearchResult]:
        """Parse numbered text responses from the MCP search server."""

        results: List[SearchResult] = []
        current: Optional[dict] = None

        def flush_current() -> None:
            if not current:
                return
            url = current.get("url", "").strip()
            title = current.get("title", "").strip()
            snippet = current.get("snippet", "").strip()
            if url and title:
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                    )
                )

        for raw_line in blob.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(r"^(\d+)\.\s+(.*)$", line)
            if match:
                if len(results) >= limit:
                    break
                flush_current()
                current = {"title": match.group(2).strip(), "snippet": ""}
                continue

            if current is None:
                continue

            if line.lower().startswith("url:"):
                current["url"] = line.split(":", 1)[1].strip()
                continue

            if line.lower().startswith("summary:"):
                current["snippet"] = line.split(":", 1)[1].strip()
                continue

            # Additional lines extend the snippet for better context.
            snippet = current.get("snippet", "")
            current["snippet"] = f"{snippet} {line}".strip()

        if len(results) < limit:
            flush_current()

        if len(results) > limit:
            return results[:limit]
        return results


class PageFetcher:
    """
    Fetch and lightly clean web pages to plain text.
    """

    def __init__(self, timeout: int = 20):
        self.headers = {"User-Agent": USER_AGENT}
        self.timeout = timeout

    def fetch(self, url: str) -> str:
        logger.debug("Fetching URL: %s", url)
        resp = requests.get(url, timeout=self.timeout, headers=self.headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # strip script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(s for s in soup.stripped_strings)
        return text


class ResearchPipeline:
    """
    Tie search + fetch together to gather evidence snippets.
    """

    def __init__(
        self,
        search_client: DuckDuckGoAPIClient,
        page_fetcher: PageFetcher,
        max_search_results: int = 5,
        max_documents: int = 3,
        planner: Optional[QueryPlanner] = None,
    ):
        self.search_client = search_client
        self.page_fetcher = page_fetcher
        self.max_search_results = max(1, max_search_results)
        self.max_documents = max(1, max_documents)
        self.planner = planner
        self.max_workers = max(1, max_documents * 2)

    def build_queries(self, company: str, address: str) -> List[str]:
        parts = [f'"{company}" "{address}"'.strip()]
        if company and address:
            parts.append(f'"{company}" "{address.split(",")[0]}" facility')
        elif company:
            parts.append(f'"{company}" facility types')
        elif address:
            parts.append(f'"{address}" facility')

        # remove duplicates and empties
        deduped: List[str] = []
        for q in parts:
            q = q.strip()
            if q and q not in deduped:
                deduped.append(q)
        logger.debug("Queries for company=%s address=%s -> %s", company, address, deduped)
        return deduped

    def collect_evidence(
        self, company: str, address: str
    ) -> Tuple[List[EvidenceDocument], QueryPlan]:
        if not company and not address:
            logger.debug("No company/address supplied for research; skipping.")
            empty_plan = QueryPlan(
                queries=[], rationale="Missing company and address", used_model=False
            )
            return [], empty_plan

        documents: List[EvidenceDocument] = []
        seen_urls: set[str] = set()

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
                    aggregated_results.extend(future.result())
        else:
            for query in queries:
                aggregated_results.extend(self._safe_search(query))

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
            return documents, plan

        fetch_workers = min(self.max_workers, max(1, self.max_documents * 2))

        if fetch_workers > 1:
            with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
                futures = [executor.submit(self._fetch_document, item) for item in fetch_candidates]
                for future in as_completed(futures):
                    if len(documents) >= self.max_documents:
                        break
                    doc = future.result()
                    if doc:
                        documents.append(doc)
        else:
            for result in fetch_candidates:
                doc = self._fetch_document(result)
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
        return documents, plan

    def _safe_search(self, query: str) -> List[SearchResult]:
        try:
            return self.search_client.search(
                query=query, max_results=self.max_search_results
            )
        except requests.RequestException as exc:
            logger.warning("Search request failed for '%s': %s", query, exc)
            return []

    def _fetch_document(self, result: SearchResult) -> Optional[EvidenceDocument]:
        try:
            content = self.page_fetcher.fetch(result.url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", result.url, exc)
            return None

        return EvidenceDocument(
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            content=content,
        )
