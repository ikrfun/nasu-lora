"""Gemma 4 E4B text-only QLoRA smoke training via Unsloth."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import torch

# Unsloth must be imported before transformers.
from unsloth import FastModel  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402


def normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora-smoke.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = tomllib.loads(Path(args.config).read_text())
    train_file = Path(config["train_file"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが利用できません")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_length"],
        load_in_4bit=True,
        dtype=None,
        text_only=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        random_state=config["seed"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def format_rows(batch: dict) -> dict[str, list[str]]:
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    normalize_messages(messages),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    dataset = dataset.map(format_rows, batched=True, remove_columns=dataset.column_names)
    print(f"Training examples: {len(dataset)}")
    print(f"Sample: {dataset[0]['text'][:160]!r}")

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        max_length=config["max_length"],
        dataset_text_field="text",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=1,
        logging_steps=1,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=config["seed"],
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        use_cache=False,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    Path(config["output_dir"], "smoke-metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
