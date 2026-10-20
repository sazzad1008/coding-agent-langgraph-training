"""CLI for isolated Git worktree runs and JSONL trace capture."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .agent import run
from .tools import default_checks


def create_worktree(repo: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="coding-agent-"))
    target.rmdir()
    result = subprocess.run(["git", "worktree", "add", "--detach", str(target), "HEAD"],
                            cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor-led coding agent")
    parser.add_argument("task")
    parser.add_argument("--workspace", type=Path, required=True, help="source Git repository")
    parser.add_argument("--model", default=os.getenv("CODING_AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--checks", type=Path, help="JSON mapping check names to argv arrays")
    parser.add_argument("--in-place", action="store_true", help="edit source repository instead of a new worktree")
    parser.add_argument("--research-steps", type=int, default=8)
    parser.add_argument("--implement-steps", type=int, default=12)
    parser.add_argument("--revisions", type=int, default=2)
    parser.add_argument("--rollout", type=Path)
    args = parser.parse_args()
    if min(args.research_steps, args.implement_steps) < 1 or args.revisions < 0:
        parser.error("step limits must be positive and revisions nonnegative")
    source = args.workspace.resolve()
    checks = json.loads(args.checks.read_text()) if args.checks else default_checks()
    if not isinstance(checks, dict) or not checks or not all(
        isinstance(name, str) and isinstance(argv, list) and argv and
        all(isinstance(part, str) and part for part in argv) for name, argv in checks.items()
    ):
        parser.error("checks must be a nonempty JSON object of argv arrays")
    workspace = source if args.in_place else create_worktree(source)
    print(f"Workspace: {workspace}")
    result = run(args.task, workspace, args.model, checks, args.research_steps,
                 args.implement_steps, args.revisions)
    print(result["answer"])
    if args.rollout:
        args.rollout.parent.mkdir(parents=True, exist_ok=True)
        with args.rollout.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"task": result["task"], "workspace": result["workspace"],
                                   "trace": result["trace"], "answer": result["answer"],
                                   "review": result["review"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
