#!/usr/bin/env python3
"""
Minimal Grok 4.1 Fast caller via OpenRouter with thinking-token stripping.

Usage:
  OPENROUTER_API_KEY=... python scripts/grok_openrouter.py --prompt "How many r's are in strawberry?"

Notes:
  - Model: x-ai/grok-4.1-fast (reasoning enabled).
  - Strips any <think>...</think> block from the content before printing "clean".
  - Prints raw content and reasoning_details for inspection.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

from openai import OpenAI


def strip_think(text: str) -> str:
    """Remove <think>...</think> block if present and return remaining text."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def call_grok(prompt: str, model: str = "x-ai/grok-4.1-fast") -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": True}},
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    clean = strip_think(raw)
    print("=== raw content ===")
    print(raw)
    print("\n=== cleaned (no <think>) ===")
    print(clean)
    print("\n=== reasoning_details ===")
    print(getattr(msg, "reasoning_details", None))


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Grok 4.1 Fast via OpenRouter with thinking-strip.")
    parser.add_argument("--prompt", required=True, help="User prompt to send.")
    parser.add_argument("--model", default="x-ai/grok-4.1-fast", help="Model id (default: x-ai/grok-4.1-fast).")
    args = parser.parse_args()
    call_grok(args.prompt, args.model)


if __name__ == "__main__":
    main()
