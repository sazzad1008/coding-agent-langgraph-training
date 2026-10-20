import json
from pathlib import Path

import pytest
from coding_agent.agent import build_graph, parse_action
from coding_agent.tools import execute, safe_path
from coding_agent.training import examples_from_rollout, prepare


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self):
        self.actions = iter([
            '{"tool":"read_file","path":"calc.py"}',
            '{"tool":"write_file","path":"calc.py","content":"def add(a, b):\\n    return a + b\\n"}',
            '{"tool":"run_tests"}',
            '{"tool":"finish","summary":"Fixed add; tests pass"}',
        ])

    def invoke(self, messages):
        return FakeResponse(next(self.actions))


def test_graph_edits_and_tests(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    state = {"task": "Fix add", "workspace": str(tmp_path), "trace": [], "action": {},
             "observation": "", "steps": 0, "max_steps": 6, "answer": ""}
    result = build_graph(FakeModel()).invoke(state)
    assert result["answer"] == "Fixed add; tests pass"
    assert "return a + b" in (tmp_path / "calc.py").read_text()
    assert "exit=0" in result["trace"][-3]["observation"]


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../secret")
    with pytest.raises(ValueError):
        execute(tmp_path, {"tool": "read_file", "path": ".env"})


def test_training_requires_verification(tmp_path):
    rollout = {"trace": [{"prompt": "p", "response": "r"}]}
    assert examples_from_rollout(rollout) == []
    rollout["verified_success"] = True
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps(rollout) + "\n")
    output = tmp_path / "train.jsonl"
    assert prepare(source, output) == 1
    assert json.loads(output.read_text()) == {"prompt": "p", "completion": "r"}


def test_parse_action():
    assert parse_action('{"tool":"list_files"}') == {"tool": "list_files"}
    with pytest.raises(ValueError):
        parse_action("[]")
