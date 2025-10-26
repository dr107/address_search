import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from requests import RequestException

from ollama import OllamaClient
from research import DuckDuckGoAPIClient, PageFetcher

logger = logging.getLogger(__name__)


@dataclass
class AgenticConfig:
    enabled: bool
    search_client: Optional[DuckDuckGoAPIClient] = None
    page_fetcher: Optional[PageFetcher] = None
    max_iterations: int = 6


class ToolAgent:
    """
    Minimal agent loop that lets a tool-capable model call
    web search and fetch utilities before producing JSON output.
    """

    def __init__(
        self,
        client: OllamaClient,
        model_name: str,
        search_client: DuckDuckGoAPIClient,
        page_fetcher: PageFetcher,
        max_iterations: int = 6,
    ):
        self.client = client
        self.model_name = model_name
        self.search_client = search_client
        self.page_fetcher = page_fetcher
        self.max_iterations = max(1, max_iterations)

    def classify(self, company_name: str, address: str) -> str:
        """
        Run the agent loop until the model emits a final answer (JSON string).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an investigative assistant that classifies business facilities. "
                    "Use the available tools to research the company/address before answering. "
                    "When you have enough evidence, respond with strict JSON:\n"
                    '{\n'
                    '  "site_type": "...",\n'
                    '  "confidence": "...",\n'
                    '  "notes": "..." \n'
                    '}\n'
                    "If evidence is insufficient, set site_type to \"unknown\" and explain why."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Company: {company}\n"
                    "Address: {address}\n"
                    "Instructions:\n"
                    "1. Call web_search to find relevant pages.\n"
                    "2. Use fetch_url on promising links to gather context.\n"
                    "3. Summarize the evidence briefly.\n"
                    "4. Return ONLY the JSON object described above."
                ).format(company=company_name or "Unknown company", address=address or "Unknown address"),
            },
        ]

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d for %s @ %s", iteration + 1, company_name, address)
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=self._tool_spec(),
                    options={
                        "temperature": 0.2,
                        "num_ctx": 4096,
                    },
                )
            except RequestException as exc:
                raise ToolAgentError(f"Ollama chat request failed: {exc}") from exc

            message = response.get("message", {})
            if not message:
                logger.warning("Agent received empty message structure from model.")
                break

            messages.append(message)
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                logger.debug("Model requested %d tool call(s).", len(tool_calls))
                for call in tool_calls:
                    tool_output = self._execute_tool(call)
                    if not tool_output:
                        tool_output = json.dumps(
                            {"error": "tool execution returned no data"},
                            ensure_ascii=False,
                        )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": call.get("function", {}).get("name"),
                        "content": tool_output,
                    }
                    messages.append(tool_message)
                continue

            final_text = self._extract_content(message.get("content"))
            if final_text:
                logger.debug("Agent produced final response.")
                return final_text

        raise ToolAgentError("Agent loop exhausted without final response.")

    def _tool_spec(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search DuckDuckGo for information about the facility.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string",
                            },
                            "count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Number of results to return",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url",
                    "description": "Fetch and summarize the textual content at a given URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Absolute URL to fetch",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def _execute_tool(self, call: Dict[str, Any]) -> str:
        function_block = call.get("function") or {}
        name = function_block.get("name")
        arguments = function_block.get("arguments") or {}

        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments else {}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}
        except json.JSONDecodeError:
            logger.warning("Tool arguments were not valid JSON: %s", arguments)
            args = {}

        if name == "web_search":
            return self._tool_web_search(args)
        if name == "fetch_url":
            return self._tool_fetch_url(args)

        logger.warning("Model requested unknown tool: %s", name)
        return json.dumps({"error": f"unknown tool '{name}'"})

    def _tool_web_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        count = int(args.get("count", 5)) if "count" in args else 5

        if not query:
            return json.dumps({"error": "web_search requires a non-empty query"})

        logger.info("Agent web_search query='%s' (count=%d)", query, count)

        try:
            results = self.search_client.search(query=query, max_results=count)
        except Exception as exc:  # broad since network errors vary
            logger.warning("web_search failed: %s", exc)
            return json.dumps({"error": f"web_search failed: {exc}"})

        payload = {
            "query": query,
            "results": [
                {"url": r.url, "title": r.title, "snippet": r.snippet} for r in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def _tool_fetch_url(self, args: Dict[str, Any]) -> str:
        url = str(args.get("url", "")).strip()
        if not url:
            return json.dumps({"error": "fetch_url requires a URL"})

        logger.info("Agent fetch_url %s", url)

        try:
            content = self.page_fetcher.fetch(url)
        except Exception as exc:
            logger.warning("fetch_url failed for %s: %s", url, exc)
            return json.dumps({"error": f"fetch_url failed: {exc}"})

        trimmed = content[:4000]
        payload = {
            "url": url,
            "content": trimmed,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict):
                    texts.append(str(block.get("text", "")))
            return "\n".join(filter(None, texts)).strip()
        return ""


class ToolAgentError(Exception):
    """Raised when the agent loop cannot complete."""
