"""Small, bounded tool surface for repository tasks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_READ = 20_000
MAX_WRITE = 50_000


def safe_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Path must be a file inside the workspace")
    if any(part in {".git", ".env", ".venv", "__pycache__"} for part in path.relative_to(root).parts):
        raise ValueError("Protected path")
    return path


def execute(root: Path, action: dict) -> str:
    tool = action.get("tool")
    if tool == "list_files":
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                       if p.is_file() and not any(x in p.relative_to(root).parts for x in (".git", ".venv", "__pycache__", ".env")))
        return "\n".join(files[:200])[:MAX_READ]
    if tool == "read_file":
        return safe_path(root, str(action["path"])).read_text(encoding="utf-8")[:MAX_READ]
    if tool == "write_file":
        path = safe_path(root, str(action["path"]))
        content = str(action["content"])
        if len(content) > MAX_WRITE:
            raise ValueError("File too large")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {path.relative_to(root)}"
    if tool == "run_tests":
        # Fixed command: model output never reaches a shell.
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root,
                                capture_output=True, text=True, timeout=60, check=False)
        return f"exit={result.returncode}\n{(result.stdout + result.stderr)[-MAX_READ:]}"
    raise ValueError(f"Unknown tool: {tool}")
