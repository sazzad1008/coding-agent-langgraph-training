"""Prepare verified rollouts and train a LoRA adapter with TRL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def examples_from_rollout(rollout: dict) -> list[dict]:
    """Only caller-approved successful rollouts become training examples."""
    if not rollout.get("verified_success", False):
        return []
    examples = []
    for entry in rollout.get("trace", []):
        if "prompt" in entry and "response" in entry:
            examples.append({"prompt": entry["prompt"], "completion": entry["response"]})
    return examples


def prepare(source: Path, destination: Path) -> int:
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as inp, destination.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            for example in examples_from_rollout(json.loads(line)):
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def train(dataset_path: Path, model_id: str, output_dir: Path, max_steps: int) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if len(dataset) == 0:
        raise ValueError("No verified training examples")
    config = SFTConfig(output_dir=str(output_dir), max_steps=max_steps,
                       per_device_train_batch_size=1, gradient_accumulation_steps=4,
                       learning_rate=2e-4, max_length=2048, completion_only_loss=True,
                       report_to="none", save_strategy="no")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    trainer = SFTTrainer(model=model_id, args=config, train_dataset=dataset, peft_config=lora)
    trainer.train()
    trainer.save_model(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("rollouts", type=Path)
    prep.add_argument("output", type=Path)
    fit = sub.add_parser("train")
    fit.add_argument("data", type=Path)
    fit.add_argument("--model", required=True)
    fit.add_argument("--output", type=Path, default=Path("checkpoints/coding-agent-lora"))
    fit.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "prepare":
        print(f"Wrote {prepare(args.rollouts, args.output)} examples")
    else:
        train(args.data, args.model, args.output, args.steps)


if __name__ == "__main__":
    main()
