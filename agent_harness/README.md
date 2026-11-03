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
