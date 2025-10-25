import argparse
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
        help="Path to input CSV. Must include a column named 'address'.",
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

    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

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
    )

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])