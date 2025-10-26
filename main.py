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

    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

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
    )

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
