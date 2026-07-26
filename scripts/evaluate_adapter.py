"""Evaluate Gemma 4 base or LoRA adapter on the fixed eval JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Unsloth must be imported before transformers.
import unsloth  # noqa: F401, E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--adapter-path", default="outputs/gemma4-e4b-qlora-full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    model_path = "google/gemma-4-E4B-it" if args.mode == "base" else args.adapter_path
    model, tokenizer = FastModel.from_pretrained(
        model_path,
        max_seq_length=512,
        load_in_4bit=True,
        text_only=True,
    )

    # Required by the current Unsloth Gemma 4 generation wrapper.
    model.config.architectures = ["Gemma4ForCausalLM"]
    if args.mode == "adapter":
        model.base_model.model.config.architectures = ["Gemma4ForCausalLM"]
    model.eval()
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    results = []
    for index, row in enumerate(load_rows(args.eval_file)):
        messages = []
        for message in row["messages"]:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append({"role": message["role"], "content": content})

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=False)
        results.append(
            {
                "index": index,
                "question": row["messages"][0]["content"],
                "response": response,
                "contains_nasu": "なす" in response or "那須" in response,
                "contains_eggplant": "🍆" in response,
                "length": len(response),
            }
        )
        print(f"{index + 1}/{len(load_rows(args.eval_file))}: {response[:100]!r}")

    summary = {
        "mode": args.mode,
        "model_path": model_path,
        "count": len(results),
        "contains_nasu_count": sum(r["contains_nasu"] for r in results),
        "contains_eggplant_count": sum(r["contains_eggplant"] for r in results),
        "average_length": sum(r["length"] for r in results) / len(results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
