#!/usr/bin/env python3
"""
Generate PII detection samples with Grok 4.1 Fast via OpenRouter.

The model is asked to emit LLaMA-Factory SFT records:
  {"instruction": "Classify PII and answer with JSON.",
   "input": "Category: <name> (<desc>). Content: <pixel_log>",
   "output": "{\"is_sensitive\": true/false, \"confidence_score\": <0-1>, \"explanation\": \"...\"}"}

Usage (macOS):
  OPENROUTER_API_KEY=... python scripts/grok_generate_dataset.py \
    --category ssn --case-type hard_pos --count 50 \
    --out data/grok_ssn_hard_pos.jsonl

Notes:
  - Model: x-ai/grok-4.1-fast
  - Reasoning enabled; strips any <think>...</think> block before parsing.
  - Saves parsed JSONL and a raw text dump alongside it (<out>.raw.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

INSTRUCTION = "Classify PII and answer with JSON."

CATEGORIES = {
    "ssn": "U.S. Social Security numbers use the pattern AAA-GG-SSSS.",
    "itin": "ITINs start with 9 and follow 9XX-7X-XXXX ranges.",
    "credit_card": "U.S. credit card numbers are 15-16 digits, often spaced or dashed.",
    "bank_account": "U.S. bank accounts are 8-17 digits; routing numbers are 9 digits.",
    "phone_number": "U.S. phone numbers have 10 digits, often with +1 and punctuation.",
    "email": "Email addresses contain a username, @, and domain.",
    "passport": "U.S. passport numbers are 9 characters, often all digits.",
    "driver_license": "U.S. driver license formats vary; CA uses 1 letter + 7 digits.",
}

CASE_TYPES = ["positive", "negative", "hard_neg", "hard_pos"]


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# Patterns/domains that indicate dummy data; filtered out after parsing.
BAN_STRINGS = {
    "123-45-6789",
    "987-65-4321",
    "123456789",
    "987654321",
    "456 78 9012",
    "shop.example.com",
    "example.com",
    "example.org",
    "example.net",
}


def build_system_prompt() -> str:
    examples = """
You are generating training data for a PII detector. 
Each sample is a Facebook Pixel-like payload formatted as a single log string that may contain U.S. PII. 
The output must be LLaMA-Factory SFT JSON objects (instruction/input/output) for classification fine-tuning.

Examples of pixel-style logs (Facebook Pixel-like, rich query payloads):
- https://www.facebook.com/tr/?id=1043363006942716&ev=ViewContent&dl=https%3A%2F%2Fuk.muji.eu%2Fproducts%2FFlannel-Cotton-Fitted-Sheet--Double-P-20740-H-000000.html&cd[content_ids]=%5B%22P-20740-H-000000%22%5D&cd[content_name]=Flannel%20Cotton%20Fitted%20Sheet%20-%20Double&pmd[description]=Cozy%20flannel%20sheet...
- https://www.facebook.com/tr/?id=969087516477356&ev=ViewContent&dl=https%3A%2F%2Fwww.zara.com%2Fus%2Fen%2Fhigh-collar-wrap-shirt-p09767673.html%3Fv1%3D502142922&cd[contents]=%5B%7B%22id%22%3A494121165%2C%22item_price%22%3A59.9%7D%5D&pmd[description]=High%20collar%20long%20sleeve%20shirt...
- https://www.facebook.com/tr/?id=1690876784663716&ev=ViewContent&dl=https%3A%2F%2Fstore.dji.com%2Fproduct%2Fosmo-360-adventure-combo%3Fvid%3D194271&pmd[keywords]=DJI%2C%20commercial%20drone%2C%20...

Output schema (one JSON object per sample, no code fences):
{
  "instruction": "Classify PII and answer with JSON.",
  "input": "Category: <name> (<desc>). Content: <pixel_log>",
  "output": "{\"is_sensitive\": true/false, \"confidence_score\": <0-1>, \"explanation\": \"<short reason>\"}"
}
- is_sensitive must be boolean true/false (not quoted).
- confidence_score between 0 and 1.
- explanation short (<120 chars), no newlines.
Keep each sample under ~250 tokens; no extra text around JSON objects.
"""
    guidance = """
Case types:
- positive: log contains valid PII of the category.
- negative: clean, no PII.
- hard_neg: looks like PII (promo codes, product slugs, SKUs, placeholders and so on) but is not.
- hard_pos: obfuscated or multilingual PII (wordified numbers, URL-encoded, at/dot emails, mixed punctuation, Spanish/Chinese/French context and others).

Requirements:
- Surface PII in varied places and vary obfuscations (wordified numbers, mixed punctuation, URL encoding, multilingual context).
- Hard cases must NOT be trivial digit-separator swaps; maximize diversity (different domains, paths, fields, languages, obfuscation styles, real-seeming product/page content and so on).
- Do not include code fences or prose—emit only JSON objects (one per line).

PII categories to cover: ssn, itin, credit_card, bank_account, phone_number, email, passport, driver_license. Hard positives should use obfuscations and multilingual context; hard negatives should look similar to PII but be non-sensitive (promo codes, SKUs, product slugs, placeholders and so on).

Final instruction on diversity:
- All examples provided above are allowed to reuse for style guidance, but do NOT limit yourself to them. Maximize diversity across domains, industries, locales, languages, obfuscation styles, and field placements in every batch. Always prefer novel, realistic variations over repeating the same patterns.

Standard log format to emit in input.content (one line):
- event=<event_type>; url=<page_url>; contents=<product info or []>; form_data=<form fields or empty>
Where:
- url may include sensitive strings if advertisers leaked them (intentional or accidental).
- contents is often benign product data (use realistic product names/descriptions/ids; good for hard negatives).
- form_data may contain PII (emails, phones, SSNs, ITINs, PANs) including obfuscations (wordified digits, at/dot emails, mixed punctuation and so on).
- Avoid obvious dummy values like 123456789, 123-45-6789, 987-65-4321, 456 78 9012, or domains like shop.example.com/example.com; use realistic-looking domains/paths (retail, travel, banking, support), varied numbers/names.
"""
    return (examples + guidance).strip()


def build_user_prompt(category: str, case_type: str, desc: str, count: int) -> str:
    return f"""
Generate {count} JSON objects in the schema above (cap generation to 10 per call).
Category: {category} ({desc})
Case type: {case_type}
Remember: one JSON object per line, no code fences, keep under ~250 tokens each; use the standard format "event=...; url=...; contents=...; form_data=..."; avoid repetitive URL+PII patterns and avoid dummy placeholders like 123456789, 123-45-6789, 987-65-4321, 456 78 9012, shop.example.com/example.com; use diverse real-ish domains/fields/obfuscations; ensure output.is_sensitive/confidence_score/explanation are valid.
""".strip()


def parse_objects(text: str) -> List[dict]:
    objs: List[dict] = []
    for m in re.finditer(r"\{.*?\}", text, re.S):
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "instruction" in obj and "input" in obj and "output" in obj:
            objs.append(obj)
    return objs


def filter_banned(objs: List[dict]) -> List[dict]:
    filtered = []
    for obj in objs:
        content = obj.get("input", "")
        if any(bad in content for bad in BAN_STRINGS):
            continue
        filtered.append(obj)
    return filtered


def call_grok(
    prompt: str, model: str = "x-ai/grok-4.1-fast"
) -> tuple[str, Optional[object]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        extra_body={"reasoning": {"enabled": True}},
        temperature=0.5,
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    return raw, getattr(msg, "reasoning_details", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PII samples with Grok via OpenRouter."
    )
    parser.add_argument("--category", required=True, choices=sorted(CATEGORIES.keys()))
    parser.add_argument("--case-type", required=True, choices=CASE_TYPES)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default="x-ai/grok-4.1-fast")
    parser.add_argument(
        "--out", required=True, help="Output JSONL path for parsed objects."
    )
    args = parser.parse_args()

    desc = CATEGORIES[args.category]
    capped = min(args.count, 10)
    user_prompt = build_user_prompt(args.category, args.case_type, desc, capped)
    raw, reasoning_details = call_grok(user_prompt, args.model)
    cleaned = strip_think(raw)
    objs = parse_objects(cleaned)
    objs = filter_banned(objs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with (out_path.with_suffix(out_path.suffix + ".raw.txt")).open(
        "w", encoding="utf-8"
    ) as f:
        f.write(raw)
        f.write("\n\n-- cleaned --\n\n")
        f.write(cleaned)
        f.write("\n\n-- reasoning_details --\n")
        f.write(json.dumps(reasoning_details, ensure_ascii=False))

    print(f"Saved {len(objs)} parsed objects to {out_path}")
    print(f"Raw content saved to {out_path.with_suffix(out_path.suffix + '.raw.txt')}")


if __name__ == "__main__":
    main()
