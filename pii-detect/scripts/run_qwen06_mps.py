#!/usr/bin/env python3
"""Evaluate the Qwen3-0.6B checkpoint locally on macOS MPS."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "qwen_eval_samples.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"
THINK_TOKEN_ID = 151668

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
    parser = argparse.ArgumentParser(description="Run Qwen3-0.6B on macOS MPS for sensitive-data eval samples.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help=f"JSONL samples file (default: {DEFAULT_SAMPLE_PATH}).",
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
        default=0.7,
        help="Sampling temperature (default: 0.7 which matches Qwen thinking-mode advice).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Force-disable Qwen thinking tokens for faster runs.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional explicit log path. Defaults to logs/qwen3_0p6b_<timestamp>.log",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict]:
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


def build_messages(sample: dict) -> list[dict]:
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
        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, content


def ensure_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend unavailable. Run on macOS with Apple Silicon and PyTorch >= 2.1.")
    return torch.device("mps")


def resolve_log_path(user_path: Path | None) -> Path:
    if user_path:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return user_path
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"qwen3_0p6b_{timestamp}.log"


def run_eval(args: argparse.Namespace, log_path: Path) -> None:
    device = ensure_mps()
    samples = load_samples(args.samples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to(device)

    thinking_flag = not args.disable_thinking

    header = (
        f"# Qwen3-0.6B evaluation\n"
        f"model: {MODEL_NAME}\n"
        f"samples: {args.samples}\n"
        f"max_new_tokens: {args.max_new_tokens}\n"
        f"temperature: {args.temperature}\n"
        f"thinking_mode: {thinking_flag}\n"
        f"device: {device}\n"
        "========================================\n"
    )
    with log_path.open("w", encoding="utf-8") as log_fh:
        print(header, file=log_fh)
        print(header, end="")

        for sample in samples:
            messages = build_messages(sample)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "temperature": max(args.temperature, 0.05),
                "do_sample": True,
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    **gen_kwargs,
                )
            thinking, content = decode_generation(tokenizer, generated_ids, model_inputs)
            expected = sample.get("expected_label", {})
            block = [f"[{sample['id']}] difficulty={sample['difficulty']} expected={expected}"]
            if thinking_flag and thinking:
                block.append(f"  thinking> {thinking}")
            block.append(f"  answer  > {content}\n")
            entry = "\n".join(block)
            print(entry, file=log_fh)
            print(entry)

    del model
    torch.mps.empty_cache()


def main() -> None:
    args = parse_args()
    log_path = resolve_log_path(args.log_file)
    run_eval(args, log_path)
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
