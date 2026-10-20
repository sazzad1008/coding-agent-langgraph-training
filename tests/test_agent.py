import json
import subprocess
import sys

import pytest
from coding_agent.agent import build_graph, parse_json
from coding_agent.cli import create_worktree
from coding_agent.tools import execute, safe_path
from coding_agent.training import examples_from_rollout, prepare


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, actions):
        self.actions = iter(actions)

    def invoke(self, messages):
        return FakeResponse(json.dumps(next(self.actions)))


def repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


def test_supervisor_research_implement_verify_review(tmp_path):
    repo(tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    model = FakeModel([
        {"plan": "Fix calc.add", "acceptance": "test passes"},
        {"tool": "read_file", "path": "calc.py"},
        {"tool": "finish", "findings": "calc.py has subtraction"},
        {"tool": "write_file", "path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
        {"tool": "finish", "summary": "Fixed calc.py"},
        {"approved": True, "findings": "Diff scoped and tests pass"},
    ])
    initial = {"task": "Fix add", "workspace": str(tmp_path), "plan": "", "acceptance": "",
               "findings": "", "implementation": "", "check_results": {}, "review": {}, "trace": [],
               "research_steps": 0, "implement_steps": 0, "revision": 0, "answer": ""}
    result = build_graph(model, {"tests": [sys.executable, "-m", "pytest", "-q"]}, 4, 4, 1).invoke(initial)
    assert result["review"]["approved"]
    assert result["check_results"]["tests"]["exit_code"] == 0
    assert "completed" in result["answer"]
    assert {x["role"] for x in result["trace"]} >= {"supervisor", "researcher", "implementer", "verifier", "reviewer"}


def test_reviewer_cannot_override_failed_check(tmp_path):
    repo(tmp_path)
    model = FakeModel([{"plan": "p", "acceptance": "a"}, {"tool": "finish", "findings": "f"},
                       {"tool": "finish", "summary": "s"}, {"approved": True, "findings": "ok"}])
    initial = {"task": "task", "workspace": str(tmp_path), "plan": "", "acceptance": "",
               "findings": "", "implementation": "", "check_results": {}, "review": {}, "trace": [],
               "research_steps": 0, "implement_steps": 0, "revision": 0, "answer": ""}
    result = build_graph(model, {"fail": ["python3", "-c", "raise SystemExit(1)"]}, 2, 2, 0).invoke(initial)
    assert not result["review"]["approved"]
    assert "incomplete" in result["answer"]


def test_path_escape_and_protected_file(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../secret")
    with pytest.raises(ValueError):
        execute(tmp_path, {"tool": "read_file", "path": ".env.local"})
    with pytest.raises(ValueError):
        execute(tmp_path, {"tool": "run_check", "name": "unconfigured"}, {"tests": ["true"]})


def test_worktree_isolated(tmp_path):
    repo(tmp_path)
    (tmp_path / "file.txt").write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    worktree = create_worktree(tmp_path)
    try:
        (worktree / "file.txt").write_text("changed")
        assert (tmp_path / "file.txt").read_text() == "original"
    finally:
        subprocess.run(["git", "-C", str(tmp_path), "worktree", "remove", "--force", str(worktree)], check=True)


def test_training_requires_verification(tmp_path):
    rollout = {"trace": [{"role": "implementer", "prompt": "p", "response": "r"}]}
    assert examples_from_rollout(rollout) == []
    rollout["verified_success"] = True
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(rollout) + "\n")
    output = tmp_path / "train.jsonl"
    assert prepare(source, output) == 1
    assert json.loads(output.read_text()) == {"prompt": "p", "completion": "r"}


def test_parse_json():
    assert parse_json('{"tool":"list_files"}') == {"tool": "list_files"}
    with pytest.raises(ValueError):
        parse_json("[]")
