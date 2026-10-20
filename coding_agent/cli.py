"""Run the agent and optionally store its observable rollout."""
import argparse
import json
import os
from pathlib import Path
from .agent import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("CODING_AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--rollout", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    result = run(args.task, args.workspace, args.model, args.max_steps)
    print(result["answer"])
    if args.rollout:
        args.rollout.parent.mkdir(parents=True, exist_ok=True)
        with args.rollout.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"task": result["task"], "trace": result["trace"],
                                   "answer": result["answer"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
