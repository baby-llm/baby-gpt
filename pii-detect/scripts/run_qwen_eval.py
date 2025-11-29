#!/usr/bin/env python3
"""Run a quick sensitivity detection eval across several Qwen3 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS: List[str] = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
]
DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "qwen_eval_samples.jsonl"
)

SYSTEM_PROMPT = (
    "You are a privacy filter that decides if a text contains U.S. sensitive "
    "information for a requested category."
)

USER_PROMPT_TEMPLATE = """Classify the content below.
Category: {name}
Description: {desc}
Content: {content}

Reply with JSON: {{"is_sensitive": <true|false>, "confidence_score": <0-1>, "explanation": "short note"}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple Qwen3 checkpoints on curated samples."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help=f"Path to the JSONL file containing evaluation samples (default: {DEFAULT_SAMPLE_PATH}).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Optional override for the list of model ids to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each answer (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0 = greedy).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Request thinking tokens (Qwen3 default).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Force-disable thinking tokens even if tokenizer supports them.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def build_messages(sample: dict) -> List[dict]:
    category = sample["category"]
    prompt = USER_PROMPT_TEMPLATE.format(
        name=category["name"],
        desc=category["desc"],
        content=sample["content"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def decode_generation(tokenizer, generated_ids, model_inputs):
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        # Qwen uses </think> token id 151668 to end intermediate reasoning blocks.
        think_token_id = 151668
        index = len(output_ids) - output_ids[::-1].index(think_token_id)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking_content, content


def run_model(
    model_name: str, samples: Iterable[dict], args: argparse.Namespace
) -> None:
    print("=" * 80)
    print(f"Evaluating {model_name}")
    print("=" * 80)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - best-effort load guard
        print(f"Failed to load {model_name}: {exc}", file=sys.stderr)
        return

    thinking_flag = True
    if args.disable_thinking:
        thinking_flag = False
    elif args.enable_thinking:
        thinking_flag = True

    for sample in samples:
        messages = build_messages(sample)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_flag,
        )
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["do_sample"] = True

        try:
            generated_ids = model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"OOM while processing {sample['id']} on {model_name}.", file=sys.stderr
            )
            torch.cuda.empty_cache()
            break
        thinking, content = decode_generation(tokenizer, generated_ids, model_inputs)
        expected = sample.get("expected_label", {})
        print(f"[{sample['id']}] difficulty={sample['difficulty']} expected={expected}")
        if thinking:
            print(f"  thinking> {thinking}")
        print(f"  answer  > {content}\n")

    # Free memory before loading next checkpoint.
    del model
    del tokenizer
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    for model_name in args.models:
        run_model(model_name, samples, args)


if __name__ == "__main__":
    main()
