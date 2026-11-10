# Swarm agent harness

A separate allocator decides how many tokens each parallel agent gets. This harness is what each agent uses to make its task cost as little as possible inside that allotment, without losing accuracy. It reports exactly what it spent.

```sh
python -m agent_harness task.json --report usage.jsonl
```

```python
from agent_harness import run_task
report = run_task({"prompt": "...", "check_cmd": "pytest -q", "workdir": "repo", "token_budget": 200000})
# report["tokens_spent"], report["budget_exhausted"], report["usage_total"], report["handoff"]
```

## Design: independent settings, start small, escalate on a failed check

There is one dispatcher. It sets each setting separately per task from cheap signals (no model call):

| Setting | Starts at | Moves up when |
|---|---|---|
| Model and effort | Sonnet low | the check fails: Sonnet high, then Opus high |
| Tools | none (lean mode) or only the task's `toolset` (`read` / `code`) | the check fails: all tools |
| Context | compacted (repetition out, never facts) | — |
| Instructions | output hygiene for Codex tool runs only | — |
| Budget | `token_budget` from the allocator | escalation stops once it is spent (`budget_exhausted`) |

Presets bundle the defaults (`config.json` → `profiles`):

| Preset | When | Behavior |
|---|---|---|
| technical | the task brings a check: `expected`, `function` + `cases`, or `check_cmd` | minimal first rung, then the full ladder |
| important | `important: true`, no check | Opus high, all tools, one call |
| general | everything else | Sonnet low, all tools, one call |

Starting small is only allowed when a check stands behind it: an under-equipped attempt fails the check and climbs to the full setup. Tasks without a check start at the safe full setup. `kind` forces a preset. If `toolset` is not given, it is inferred: `code` (Bash, Read, Edit, Write) when there is a workdir or check command, otherwise `none`.

## Measured (2026-10-29, Claude Code 2.1.314, Codex 0.164.0, Sonnet low unless noted)

| What | Result |
|---|---|
| Fixed input on a trivial question | All Claude tools: 20,933. Bash, Read, Edit, Write: 6,031. Read: 3,905. None: 2,814. None plus a short system prompt: 579. Codex default: 16,940. |
| Lean first rung, tool-free tasks (`pilot_v2`, 12 hard tasks) | Default 12/12 at 21,134 tokens per correct. Lean 11/12 at 847 tokens per correct; its miss was caught by the check. Probably overstated: each run used a fresh folder, which breaks the cache and inflates the default's cost. |
| Toolset first rung, easy tool tasks (`pilot_toolset_v1`, 3 tasks × 2) | Full tools 6/6 at 134,664 tokens per correct. Code toolset 6/6 at 30,369 tokens per correct (−77%). |
| Toolset first rung, recheck (`recheck_toolset_v1`, 3 multi-file projects × 2, hidden tests) | Full tools 6/6, code toolset 6/6. Turns 33 vs 34, so the saving is per-turn tool overhead. **Tokens −70%; Claude's list-price cost −60%.** Reported usage matched the per-model totals on every run. Treat about 60–70% as the realistic range; 12 pairs cannot rule out a loss rate of up to about 22%, so the check-backed fallback still matters. |
| Output hygiene (`pilot_v2`, 3 tasks × 2) | Codex 6/6 both ways, −36% tokens per correct: on. Claude 6/6 both ways, +4%: off. |
| Compaction (no model call) | Noisy logs −80.6% with zero facts lost. Prose about 0%. It falls back to the original if any fact would be lost or the text would grow. |
| End to end through the CLI | All three presets answered correctly. Technical solved on the lean rung with 619 input tokens. |

## Limits

- Escalation and minimal rungs are only as safe as the check. Prefer real test suites; weak tests can accept a wrong cheap answer.
- Every test task was written by the person testing the ideas (graded by hidden tests where noted, nothing tuned to them). All runs were Sonnet low on small Python projects. Tasks needing web, search or sub-agent tools would fail the minimal rung and pay for an extra attempt before falling back.
- The pilots are small (12 lean tasks; 3 tool tasks × 2 repeats; 3 multi-file projects × 2). Agent runs vary run to run (one full-tools run used 227k tokens where its repeat used 115k). Re-run `python -m agent_harness.pilot --parts ABC` after CLI or model updates.
- How cached tokens count against subscription limits is not documented in what we found. Reports list every token category so the allocator can weight them.
- Codex has no verified way to restrict built-in tools or instructions, so it gets no minimal rung. Hygiene is its saving.
- General and important tasks keep Claude's full fixed overhead, because there is no check to catch a minimal-setup mistake. Lowering it would need a retention-gated pilot.
- Budget stops apply between attempts. A single attempt can still exceed the remaining budget.

## Failure handling and rung isolation (2026-11-10)

- A timed-out or failed `claude`/`codex` run is a failed attempt, never a crash. Usage it printed before failing is still counted, so budgets don't undercount when calls fail and escalate. Attempts carry `timed_out` when that happened.
- A hung `check_cmd` fails the check (`check_timed_out`). If a runner or check raises unexpectedly, the ladder stops, and the report keeps every earlier attempt plus an `exception` field.
- Checked tasks with a `workdir` and more than one rung snapshot the workdir first and restore it before each escalation, so every rung starts from the same files (`workspace.py`). `.git`, `node_modules`, virtualenvs and caches are not copied or restored. The cost is reported in `rung_isolation`. Settings: `isolate_rungs` (default true) and `isolate_rungs_max_mb` (default 100; a larger workdir runs unisolated and says so). Restoring deletes files created during the failed rung, so don't share one workdir between concurrent tasks.
- `compact_checked` only rescans lines that compaction changed when checking for lost facts. The result is identical to the full scan (property-tested). On 10,000 unique lines it takes 15 ms instead of 121 ms.

## General tasks (2026-11-10)

```sh
python -m agent_harness --prompt "Fix the failing test" --workdir repo --check-cmd "pytest -q"
python -m agent_harness --prompt "Extract the invoice fields" --context-file invoice.txt --schema invoice.schema.json
python -m agent_harness tasks.jsonl --jobs 8 --report usage.jsonl     # a batch: JSON list or JSON Lines
```

Exit code: 0 if everything was verified or unchecked, 1 if a check failed, 2 if an exception happened.

| Task key | What it does |
|---|---|
| `expected`, `function`+`cases`, `check_cmd` | **Correctness checks.** Only these route a task to the technical ladder and its cheap first rung. |
| `regex`, `contains`, `output_schema`, `checks: [{"type": ...}]` | **Format checks.** The task keeps its normal full setup (no cheap rung, because a format check can't catch a wrong answer). If the format is wrong, it climbs to stronger rungs. `output_schema` is also passed to Claude (`--json-schema`) and Codex (`--output-schema`). All checks must pass; cheap ones run first. An unknown check type or schema keyword fails the check instead of passing it. |
| `context_files` | Files included verbatim. Only free-form `context` is compacted. |
| `timeout`, `check_timeout` | Seconds per agent call and per check. |

- **Transient API errors** (overloaded, rate limits, 5xx, network) retry the same rung up to 2 times, waiting 10 s and then 30 s (`retry` in config), instead of escalating to a pricier model. Retries are counted in usage and the budget. The harness's own timeout is never retried.
- **Batches** run in parallel threads. Tasks that share a `workdir` run one after another.
- **The Codex sandbox follows the toolset:** `code` gets `workspace-write`, everything else `read-only`. The old fixed read-only setting meant a Codex coding rung could never pass.
- The research graders (`independent_audit`) load only when `expected` or `cases` checks are used.
- **Live check** (Claude Code 2.1.322): `--json-schema` works on the tool-free lean rung. A schema task answered in 1,126 input tokens and 2.1 s.
- **Permissions:** headless Claude can't ask for approval, so the `code` and `full` toolsets get `--permission-mode acceptEdits` (edits inside the working folder only; `permissions` in config). Shell commands stay opt-in with `allowed_tools` / `--allow "Bash(pytest*)"`. The harness never adds the check command itself, so hidden tests stay hidden. Before this, a coding task through the default config couldn't edit anything; the pilots only worked because they passed their own `--allowedTools`.
- **Hard spend limit:** `max_budget_usd` / `--max-usd` gives each Claude call whatever is left (`--max-budget-usd`), and the ladder stops when it reaches zero. Claude Code checks the limit *after* each model turn, so one turn can overshoot: a $0.0001 limit still spent $0.016–0.048. Codex has no equivalent. `token_budget` still only reports.
- **Usage fallback:** when a run stops on the spend limit, Claude Code reports zero `usage` but still fills in `modelUsage`. The harness falls back to `modelUsage`, so these tokens are counted.
- **Live end-to-end** (2026-11-10, Claude Code 2.1.322, plain default config plus `allowed_tools: ["Bash(python3 -m unittest*)"]`): the `inventory` multi-file bug fix passed its hidden tests on the code-toolset rung. 44k input tokens, 16.8 s, workdir snapshot 2 ms.
