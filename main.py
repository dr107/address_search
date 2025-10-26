import argparse
import logging
import sys
from pathlib import Path
from typing import List

from csvparse import process_file


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a CSV of addresses using a local Ollama model (scaffold)."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV. Must include columns 'Company Name' and 'Full Address'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write output CSV.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama server. Default: http://localhost:11434",
    )
    parser.add_argument(
        "--model",
        default="llama3.1-8b-instruct",
        help="Model name/tag to use in Ollama. Default: llama3.1-8b-instruct",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process (useful for dry runs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO",
    )
    parser.add_argument(
        "--enable-web-research",
        action="store_true",
        help="If set, run DuckDuckGo search + fetch to gather evidence.",
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=5,
        help="Number of search results to inspect per query (default: 5).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=3,
        help="Maximum number of web documents to fetch per row (default: 3).",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for search/fetch HTTP requests (default: 20).",
    )
    parser.add_argument(
        "--search-api-url",
        default="http://localhost:8000",
        help="Base URL for the local DuckDuckGo OpenAPI server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--agentic-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control whether to use the tool-calling agent loop (default: auto).",
    )
    parser.add_argument(
        "--agentic-model-hints",
        default="llama3,llama-3,llama4,llama-4,deepseek",
        help="Comma-separated substrings that should trigger agentic mode when --agentic-mode=auto.",
    )
    parser.add_argument(
        "--agent-max-iterations",
        type=int,
        default=6,
        help="Maximum tool-call iterations when agentic mode is enabled (default: 6).",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help=(
            "If set, recompute every row even when the output CSV already contains prior results."
        ),
    )

    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    agentic_model_hints = [
        hint.strip()
        for hint in (args.agentic_model_hints or "").split(",")
        if hint.strip()
    ]

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    process_file(
        input_path=input_path,
        output_path=output_path,
        ollama_url=args.ollama_url,
        model_name=args.model,
        limit=args.limit,
        enable_web_research=args.enable_web_research,
        max_search_results=args.max_search_results,
        max_documents=args.max_documents,
        fetch_timeout=args.fetch_timeout,
        search_api_url=args.search_api_url,
        agentic_mode=args.agentic_mode,
        agentic_model_hints=agentic_model_hints,
        agent_max_iterations=args.agent_max_iterations,
        ignore_cache=args.ignore_cache,
    )

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
