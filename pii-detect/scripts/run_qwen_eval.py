#!/usr/bin/env python3
"""Evaluate multiple Qwen checkpoints on curated PII detection samples.

The script is defensive against tokenizer API changes (e.g., conversation vs.
messages argument) and can optionally log results to JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS: List[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3-8B",
]
DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "qwen_eval_samples.jsonl"
)

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "eval_results" / "result.json"
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


@dataclass
class Sample:
    id: str
    difficulty: str
    category: dict
    content: str
    expected_label: Optional[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple Qwen checkpoints on curated samples."
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
    parser.add_argument(
        "--device-map",
        default="auto",
        help='device_map passed to HF model loading (default: "auto").',
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="torch dtype for model weights (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write combined JSON results (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable writing results to disk; only print to stdout.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop evaluation if a sample/model raises an unexpected error.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading models/tokenizers.",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def load_samples(path: Path) -> List[Sample]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append(
                Sample(
                    id=raw["id"],
                    difficulty=raw.get("difficulty", "unknown"),
                    category=raw["category"],
                    content=raw["content"],
                    expected_label=raw.get("expected_label"),
                )
            )
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def detect_thinking_support(tokenizer) -> tuple[bool, Optional[int]]:
    """Check whether the tokenizer supports thinking mode and return the token id."""
    try:
        sig = signature(tokenizer.apply_chat_template)
        has_param = "enable_thinking" in sig.parameters
    except (TypeError, ValueError):
        has_param = False

    think_token_id = None
    if has_param:
        try:
            token_id = tokenizer.convert_tokens_to_ids("</think>")
        except Exception:
            token_id = None
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk_id:
            think_token_id = token_id
        else:
            has_param = False

    return has_param, think_token_id


def build_messages(sample: Sample) -> List[dict]:
    prompt = USER_PROMPT_TEMPLATE.format(
        name=sample.category["name"],
        desc=sample.category["desc"],
        content=sample.content,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def apply_chat_template_safe(
    tokenizer,
    messages: List[dict],
    enable_thinking: bool,
    add_generation_prompt: bool = True,
) -> str:
    """Apply chat template across tokenizer versions."""
    try:
        sig = signature(tokenizer.apply_chat_template)
        has_thinking_param = "enable_thinking" in sig.parameters
    except (TypeError, ValueError):
        has_thinking_param = False

    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if has_thinking_param:
        kwargs["enable_thinking"] = enable_thinking
    elif enable_thinking:
        print(
            "Tokenizer does not expose enable_thinking; continuing without it.",
            file=sys.stderr,
        )

    # Try common call signatures in order of modern APIs.
    attempts = [
        {"args": (messages,), "kwargs": kwargs},
        {"args": (), "kwargs": {"conversation": messages, **kwargs}},
        {"args": (), "kwargs": {"messages": messages, **kwargs}},
    ]
    last_exc: Optional[Exception] = None
    for attempt in attempts:
        try:
            return tokenizer.apply_chat_template(*attempt["args"], **attempt["kwargs"])
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"tokenizer.apply_chat_template failed with known signatures: {last_exc}"
    )


def decode_generation(
    tokenizer,
    generated_ids,
    model_inputs,
    think_token_id: Optional[int] = None,
) -> tuple[str, str]:
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    index = 0
    if think_token_id is not None:
        try:
            # Qwen uses </think> token to end intermediate reasoning blocks.
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            index = 0
    thinking_content = (
        tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        if index
        else ""
    )
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking_content, content


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def load_model_and_tokenizer(model_name: str, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_for_sample(
    model,
    tokenizer,
    sample: Sample,
    thinking_flag: bool,
    think_token_id: Optional[int],
    args: argparse.Namespace,
) -> dict:
    messages = build_messages(sample)
    text = apply_chat_template_safe(
        tokenizer, messages, enable_thinking=thinking_flag
    )
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["do_sample"] = True

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **gen_kwargs,
        )

    thinking, content = decode_generation(
        tokenizer,
        generated_ids,
        model_inputs,
        think_token_id if thinking_flag else None,
    )
    return {
        "sample_id": sample.id,
        "difficulty": sample.difficulty,
        "expected": sample.expected_label,
        "thinking": thinking,
        "answer": content,
    }


def run_model(
    model_name: str, samples: Iterable[Sample], args: argparse.Namespace
) -> List[dict]:
    print("=" * 80)
    print(f"Evaluating {model_name}")
    print("=" * 80)
    if "Instruct" not in model_name and "Chat" not in model_name:
        print(
            f"Note: {model_name} is a base checkpoint; instruction following may be weak. Prefer an *-Instruct variant.",
            file=sys.stderr,
        )
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, args)
    except Exception as exc:  # pragma: no cover - best-effort load guard
        print(f"Failed to load {model_name}: {exc}", file=sys.stderr)
        return []

    supports_thinking, think_token_id = detect_thinking_support(tokenizer)
    thinking_flag = supports_thinking
    if args.disable_thinking:
        thinking_flag = False
    elif args.enable_thinking and supports_thinking:
        thinking_flag = True
    elif args.enable_thinking and not supports_thinking:
        print(
            f"{model_name} does not expose thinking tokens; continuing with thinking disabled.",
            file=sys.stderr,
        )

    results: List[dict] = []
    for sample in samples:
        try:
            result = generate_for_sample(
                model,
                tokenizer,
                sample,
                thinking_flag,
                think_token_id,
                args,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"OOM while processing {sample.id} on {model_name}.", file=sys.stderr
            )
            torch.cuda.empty_cache()
            break
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(
                f"Error while processing {sample.id} on {model_name}: {exc}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise
            continue

        expected = result["expected"] or {}
        print(
            f"[{sample.id}] difficulty={sample.difficulty} expected={expected}"
        )
        if result["thinking"]:
            print(f"  thinking> {result['thinking']}")
        print(f"  answer  > {result['answer']}\n")
        results.append({"model": model_name, **result})

    # Free memory before loading next checkpoint.
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return results


def append_results(path: Path, results: List[dict]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    all_results: List[dict] = []
    for model_name in args.models:
        results = run_model(model_name, samples, args)
        all_results.extend(results)
    if not args.no_save and all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
