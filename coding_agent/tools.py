"""Repository tools with bounded output and no model-generated shell commands."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAX_OUTPUT = 24_000
MAX_WRITE = 100_000
PROTECTED = {".git", ".env", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


def safe_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Path must be a file inside the workspace")
    if any(part in PROTECTED or part.startswith(".env.") for part in path.relative_to(root).parts):
        raise ValueError("Protected path")
    return path


def command(root: Path, argv: list[str], timeout: int = 90) -> dict:
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                                timeout=timeout, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return {"exit_code": result.returncode, "output": (result.stdout + result.stderr)[-MAX_OUTPUT:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "output": f"Timed out after {timeout}s"}
    except OSError as exc:
        return {"exit_code": 127, "output": str(exc)}


def execute(root: Path, action: dict, checks: dict[str, list[str]] | None = None) -> dict:
    """Execute one validated action. Check names map to operator-configured argv."""
    root = root.resolve()
    tool = action.get("tool")
    if tool == "list_files":
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                       and not any(x in PROTECTED or x.startswith(".env.") for x in p.relative_to(root).parts))
        return {"exit_code": 0, "output": "\n".join(files[:500])[:MAX_OUTPUT]}
    if tool == "search":
        pattern = str(action["pattern"])
        if len(pattern) > 200:
            raise ValueError("Search pattern too long")
        return command(root, ["rg", "-n", "--max-count", "20", "--glob", "!.env*", "--glob", "!.git/**",
                              "--glob", "!node_modules/**", "--", pattern, "."], timeout=15)
    if tool == "read_file":
        return {"exit_code": 0, "output": safe_path(root, str(action["path"])).read_text(encoding="utf-8")[:MAX_OUTPUT]}
    if tool == "write_file":
        path = safe_path(root, str(action["path"]))
        content = str(action["content"])
        if len(content) > MAX_WRITE:
            raise ValueError("File too large")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"exit_code": 0, "output": f"Wrote {path.relative_to(root)}"}
    if tool == "git_diff":
        return command(root, ["git", "diff", "--", "."], timeout=15)
    if tool == "git_status":
        return command(root, ["git", "status", "--short"], timeout=15)
    if tool == "run_check":
        name = str(action["name"])
        if checks is None or name not in checks:
            raise ValueError(f"Unknown configured check: {name}")
        argv = checks[name]
        if not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError("Invalid check argv")
        return command(root, argv, timeout=120)
    raise ValueError(f"Unknown tool: {tool}")


def default_checks() -> dict[str, list[str]]:
    return {"tests": [sys.executable, "-m", "pytest", "-q"]}
