#!/usr/bin/env python
"""Route a query and optional code through the baseline routing layer."""

import argparse
import json
from pathlib import Path

from routing import route_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="User request to route")
    parser.add_argument("--code-file", type=Path)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--expected-output-tokens", type=int, default=1200)
    parser.add_argument("--quality-threshold", type=float, default=0.75)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    code = (
        args.code_file.read_text(encoding="utf-8")
        if args.code_file is not None
        else None
    )
    decision = route_request(
        args.query,
        code,
        context_tokens=args.context_tokens,
        expected_output_tokens=args.expected_output_tokens,
        quality_threshold=args.quality_threshold,
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
