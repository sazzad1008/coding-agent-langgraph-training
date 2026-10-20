# Supervisor-led coding agent with LangGraph

A repository coding workflow with distinct supervisor, researcher, implementer, verifier, and reviewer roles. It uses LangChain for the chat model and LangGraph for bounded role transitions, then records observable rollouts for verified supervised fine-tuning. The default run creates a detached Git worktree, so the source checkout stays untouched.

```text
request → supervisor plan → researcher tools → implementer tools
                                                ↓
         final report ← reviewer ← configured checks + Git diff/status
                           ↓ if rejected (bounded)
                        implementer
```

## What each role does

| Role | Responsibility | Allowed tools |
| --- | --- | --- |
| Supervisor | Define plan and acceptance criteria | None |
| Researcher | Find relevant files and tests | List, search, read, Git status |
| Implementer | Make scoped changes | Read, search, write file, Git diff |
| Verifier | Run every configured check and capture diff/status | Operator-defined argv only |
| Reviewer | Inspect check results and changes; approve or request repair | None |

The reviewer cannot turn a failed check into success. A failed or exhausted run reports `incomplete`. This is a coordinated multi-role workflow, not independent processes running in parallel.

## Install

Python 3.11+ and Git are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
export OPENAI_API_KEY='your-key'
```

Set `CODING_AGENT_MODEL` to change the model. `OPENAI_BASE_URL` can point ChatOpenAI to a compatible endpoint. The agent requires an existing Git repo with at least one commit for worktree creation.

## Run on a repository

```bash
coding-agent 'Fix the failing authentication test and add a regression test' \
  --workspace /absolute/path/to/your/repo \
  --checks /absolute/path/to/checks.json \
  --rollout runs/rollouts.jsonl
```

The CLI prints the new worktree path. Review its diff with `git -C <worktree> diff` and `git -C <worktree> status --short`. Changes stay there for review. Use `--in-place` only when you intentionally want edits in the source checkout.

`--checks` is a JSON mapping of check names to argument arrays. Example: [checks.example.json](checks.example.json). These commands are chosen by the operator and run inside the agent worktree. Adapt them to the target repository's test, lint, typecheck, or build commands. If omitted, the default runs `python -m pytest -q` using the agent's Python interpreter. The agent cannot invent shell commands through its tool calls.

Tune budgets with `--research-steps`, `--implement-steps`, and `--revisions`. Defaults are 8, 12, and 2. The verifier reruns all checks after each implementation round.

## Training pipeline

The rollout file stores model prompts, responses, tool results, review, and outcome. It may contain repository code or sensitive output: treat it as private data. An independent evaluator should run held-out tests and inspect the patch. Add `"verified_success": true` only to rollouts that genuinely satisfy the task; the agent's own review does not set this flag.

```bash
python -m coding_agent.training prepare runs/rollouts.jsonl runs/train.jsonl
pip install -e '.[train]'
python -m coding_agent.training train runs/train.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output checkpoints/coding-agent-lora \
  --steps 100
```

Only implementer decisions become prompt/completion examples. Tool outputs appear as context and are not prediction targets. The result is a PEFT LoRA adapter. Serve it with its base model through an OpenAI-compatible endpoint, then measure held-out task resolution, regressions, and cost against the base model. Training requires suitable accelerator memory and a meaningful set of verified tasks.

## Checks and boundaries

```bash
pytest -q
```

The included offline tests cover role routing, failed-check rejection, path protection, worktree isolation, and training-data gating. GitHub Actions runs them on pushes and pull requests.

The file tools block paths outside the worktree and common secret/dependency directories, and tool output and steps are bounded. **This is not a secure sandbox.** Configured checks and repository tests execute code. Run untrusted repositories in a locked-down container or VM. The agent does not commit, merge, or push task changes automatically. Human review remains necessary before shipping patches.
