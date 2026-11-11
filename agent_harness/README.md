# Swarm agent harness

A separate allocator decides how many tokens each parallel agent gets. This harness is what each agent uses to make its task cost as little as possible inside that allotment, without losing accuracy. It reports exactly what it spent.

```sh
python -m agent_harness task.json --report usage.jsonl
```

```python
from agent_harness import run_task
report = run_task({"prompt": "...", "check_cmd": "pytest -q", "workdir": "repo", "token_budget": 200000})
# report["tokens_spent"], report["budget_exhausted"], report["usage_total"], report["handoff"]
# usage_total also splits new_input / cached_input and gives an illustrative weighted_tokens
# (new + 0.1 × cached + 8 × output, config report_weights): cache re-reads are cheap, so a
# run can use more total tokens and still cost less. Budgets use total tokens.
```

## Design: independent settings, start small, escalate on a failed check

There is one dispatcher. It sets each setting separately per task from cheap signals (no model call):

| Setting | Starts at | Moves up when |
|---|---|---|
| Model and effort | Sonnet low | the check fails: Sonnet high, then Opus high |
| Tools | Claude: none (lean mode) or only the task's `toolset` (`read` / `code`). Codex: no web search or plugins | the check fails: all tools |
| Context | compacted (repetition out, never facts) | — |
| Instructions | none by default (output hygiene is opt-in per agent: `hygiene_agents`) | — |
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
| Output hygiene (`pilot_v2`, 3 tasks × 2) | Codex 6/6 both ways, −36% tokens per correct. Claude 6/6 both ways, +4%. |
| Output hygiene, Codex, Lambda's counterbalanced A/B (2026-11-10, 5 coding tasks × 3 pairs, alternating order) | 15/15 both ways. With hygiene: total tokens +5.3%, weighted +8.0%, time +10.5% (per task −18% to +43%). The pilot saving did not replicate: **off by default**. |
| **Codex toolset first rung, live A/B** (`ab_codex_toolset_v1`, 2026-11-10, codex-cli 0.164.0, installed default model; 6 coding tasks × 3 pairs, alternating order, hidden tests) | Plain Codex 18/18, harness 18/18, no fallbacks needed. **Tokens −32.7%** (fewer in 18/18 pairs), weighted −29.0% (15/18), time −19.1% (15/18). Per task, weighted: CSV −54%, exprcalc −37%, slots −31%, tests −25%, inventory −23%, log 0%. Fixed input per model turn 16,926 → 11,549 (web search 2,456, plugins 2,006). `python -m agent_harness.ab_codex` reruns it. |
| Compaction (no model call) | Noisy logs −80.6% with zero facts lost. Prose about 0%. It falls back to the original if any fact would be lost or the text would grow. |
| End to end through the CLI | All three presets answered correctly. Technical solved on the lean rung with 619 input tokens. |

## Limits

- Escalation and minimal rungs are only as safe as the check. Prefer real test suites; weak tests can accept a wrong cheap answer.
- Every test task was written by the person testing the ideas (graded by hidden tests where noted, nothing tuned to them). All runs were Sonnet low on small Python projects. Tasks needing web, search or sub-agent tools would fail the minimal rung and pay for an extra attempt before falling back.
- The pilots are small (12 lean tasks; 3 tool tasks × 2 repeats; 3 multi-file projects × 2). Agent runs vary run to run (one full-tools run used 227k tokens where its repeat used 115k). Re-run `python -m agent_harness.pilot --parts ABC` after CLI or model updates.
- How cached tokens count against subscription limits is not documented in what we found. Reports list every token category so the allocator can weight them.
- Codex's minimal rung only drops web search and plugins (`agents.codex.toolset_args`); its built-in instructions stay. Output hygiene (a "keep output short" instruction, the approach RTK also uses for Codex) measured worse in Lambda's larger A/B and is off. Asking the model to be brief changes how it works; removing tools it never calls doesn't.
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

- **Transient API errors** (overloaded, rate limits, 5xx, network) retry the same rung once after 5 s (`retry` in config), instead of escalating to a pricier model. Claude Code already retries internally before it exits, so a longer wait would mostly add lag. Retries are counted in usage and the budget. The harness's own timeout is never retried.
- **Batches** run in parallel threads. Tasks that share a `workdir` run one after another.
- **The Codex sandbox follows the toolset:** `code` gets `workspace-write`, everything else `read-only`. The old fixed read-only setting meant a Codex coding rung could never pass.
- The research graders (`independent_audit`) load only when `expected` or `cases` checks are used.
- **Live check** (Claude Code 2.1.322): `--json-schema` works on the tool-free lean rung. A schema task answered in 1,126 input tokens and 2.1 s.
- **Permissions:** headless Claude can't ask for approval, so the `code` and `full` toolsets get `--permission-mode acceptEdits` (edits inside the working folder only; `permissions` in config). Shell commands stay opt-in with `allowed_tools` / `--allow "Bash(pytest*)"`. The harness never adds the check command itself, so hidden tests stay hidden. Before this, a coding task through the default config couldn't edit anything; the pilots only worked because they passed their own `--allowedTools`.
- **Hard spend limit:** `max_budget_usd` / `--max-usd` gives each Claude call whatever is left (`--max-budget-usd`), and the ladder stops when it reaches zero. Claude Code checks the limit *after* each model turn, so one turn can overshoot: a $0.0001 limit still spent $0.016–0.048. Codex has no equivalent. `token_budget` still only reports.
- **Usage fallback:** when a run stops on the spend limit, Claude Code reports zero `usage` but still fills in `modelUsage`. The harness falls back to `modelUsage`, so these tokens are counted.
- **Live end-to-end** (2026-11-10, Claude Code 2.1.322, plain default config plus `allowed_tools: ["Bash(python3 -m unittest*)"]`): the `inventory` multi-file bug fix passed its hidden tests on the code-toolset rung. 44k input tokens, 16.8 s, workdir snapshot 2 ms.

## Overhead (2026-11-10)

Harness time per task with an instant fake model, so the model call itself isn't counted (`40c1812` → this branch):

| Case | Before | Now |
|---|---|---|
| General task / exact-answer task | 0.02–0.05 ms | same |
| General task with a 2k-line log | 28 ms | 6 ms |
| Coding task with check, 10 files | 7 ms | 11 ms |
| Coding task with check, 5k files / 50 MB | 8 ms | 111 ms (snapshot) |
| Same, all 4 rungs fail (3 restores) | 27 ms | 313 ms |
| CLI process, start to finish | 67 ms | 64 ms |

How rung isolation stays cheap:
- **macOS:** one APFS `clonefile()` per top-level entry clones whole folders in the kernel (5k files: 36 ms, vs 727 ms cloning file by file).
- **Linux:** `cp -a --reflink=auto` is used where possible. Plain ext4 has no reflink, so it falls back to a native-speed copy; expect more than on macOS.
- **Restore** compares stat fingerprints and rewrites only the files that changed.
- **Snapshot deletion** runs in a background thread.

Import cost: the research graders, `concurrent.futures` and `argparse` load only when a check, a batch or the CLI needs them.

## Chatbot use: same speed, fewer tokens (2026-11-10, live, median of 3, Sonnet low)

| Setup | Done | First text | Input tokens |
|---|---|---|---|
| `claude -p` direct, no harness | 2.73 s | at the end | 21,058 |
| `run_task` in-process | 2.60 s | at the end | 21,058 |
| `run_task(..., on_text=cb)` | 2.31 s | **0.98 s** | 21,058 |
| same, task declares `"toolset": "none"` | 2.35 s | 0.91 s | **2,913** |

- **Speed:** called in-process, the harness runs at the same speed as calling Claude directly. Launched as a CLI, it adds about 0.1–0.2 s of Python startup.
- **Streaming:** pass `on_text` to stream the answer as it's generated. This applies to unchecked tasks only, because a checked rung's answer can still fail its check and be replaced.
- **Declared toolset:** on unchecked tasks the harness now honours a toolset the caller *declares*. Claude Code's normal system prompt stays; only the unneeded tools are dropped. A toolset the harness only infers is still never applied to unchecked tasks.
- **Not worth it:** pre-starting `claude` with stream-json input (1.71 → 1.66 s, and it used 2k more tokens). MCP connectors cost no tokens because their tools load on demand. Claude Code's own startup (about 0.9 s of each call, against about 0.9 s of API time) only goes away if you call the API directly instead of the CLI.

## Platforms (2026-11-10)

Lambda's audit ran V2's tests on Windows: 40 passed, 15 failed. Fixes:
- **Function-case checks could never pass on Windows.** The sandboxed grader imported `resource`, which Windows doesn't have, so every `function`+`cases` task failed on every rung. It now applies resource limits only where they exist; the 4 s timeout still applies everywhere.
- **Restore fingerprints on Windows.** `st_ctime` there is the creation time, so an equal fingerprint doesn't prove a file is unchanged. Windows confirms by comparing content, which only happens during escalation. On POSIX, ctime remains proof. A test covers an edit that puts back the original mtime, on both paths.
- `command` may be a list (`["npx", "claude"]`), and bare names resolve through PATH/PATHEXT (`claude.cmd`, `claude.exe`). The hygiene hint uses `python` on Windows.
- **No torch needed.** The harness, its pilots and its tests no longer import `torch` (they used to, via `independent_audit.model`, just for a path).
- The tests are cross-platform (no `/bin/sh`, `true`, `grep` or shebangs), and CI runs them plus `python -m agent_harness.bench` on Ubuntu, macOS and Windows (`.github/workflows/harness-tests.yml`).
