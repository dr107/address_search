from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

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
        base_url: str = "http://localhost:8001",
        timeout: int = 15,
        safe_search: str = "moderate",
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.safe_search = safe_search

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        logger.debug(
            "Searching DuckDuckGo API for query=%s (count=%d)", query, max_results
        )
        payload = {
            "query": query,
            "count": max(1, min(20, max_results)),
            "safeSearch": self.safe_search,
        }
        resp = self.session.post(
            f"{self.base_url}/duckduckgo_web_search",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        items: Optional[List[dict]] = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                items = data["results"]
            elif "data" in data and isinstance(data["data"], list):
                items = data["data"]

        results: List[SearchResult] = []
        if not items:
            logger.debug("DuckDuckGo API returned no parsable results for %s", query)
            return results

        for item in items:
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


class PageFetcher:
    """
    Fetch and lightly clean web pages to plain text.
    """

    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout

    def fetch(self, url: str) -> str:
        logger.debug("Fetching URL: %s", url)
        resp = self.session.get(url, timeout=self.timeout)
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
    ):
        self.search_client = search_client
        self.page_fetcher = page_fetcher
        self.max_search_results = max(1, max_search_results)
        self.max_documents = max(1, max_documents)

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

    def collect_evidence(self, company: str, address: str) -> List[EvidenceDocument]:
        if not company and not address:
            logger.debug("No company/address supplied for research; skipping.")
            return []

        documents: List[EvidenceDocument] = []
        seen_urls: set[str] = set()

        for query in self.build_queries(company, address):
            try:
                results = self.search_client.search(
                    query=query, max_results=self.max_search_results
                )
            except requests.RequestException as exc:
                logger.warning("Search request failed for '%s': %s", query, exc)
                continue

            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)

                try:
                    content = self.page_fetcher.fetch(result.url)
                except requests.RequestException as exc:
                    logger.warning("Failed to fetch %s: %s", result.url, exc)
                    continue

                documents.append(
                    EvidenceDocument(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        content=content,
                    )
                )

                if len(documents) >= self.max_documents:
                    logger.debug(
                        "Reached max_documents=%d; stopping evidence collection.",
                        self.max_documents,
                    )
                    return documents

        logger.debug(
            "Collected %d evidence document(s) for company=%s address=%s",
            len(documents),
            company,
            address,
        )
        return documents
