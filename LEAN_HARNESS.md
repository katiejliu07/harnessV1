# Token-light runtime

The new entry point is `lean_harness`; the original repository and completed
independent audit remain unchanged. The prompt benchmark is currently running.
`lean_harness/policy.json` is written only after development, fresh validation
and the original regression suite finish. Failed candidates remain in the results.

```sh
cd /Users/katieliu/Downloads/harness
independent_audit/.venv/bin/python -m lean_harness --kind code --prompt "Write a Python function dedup(items) that preserves first occurrences and input order. Items are hashable. Handle empty input without modifying it."
independent_audit/.venv/bin/python -m lean_harness --kind scalar --prompt "What is 13/16 of 240?"
```

Use `--kind general` for requests that need explanations, caveats, structured
prose or other output beyond a number/function. General requests pass through
unchanged. Use `--prompt-file` for full tasks including tests and constraints.
`--metadata result.json` retains tokens, raw text, model settings and measured
request time. EOS does not prove correctness; supply real tests for generated code.

The runtime removes redundant prompt wording, donor computation and duplicate
tokenization. It never strips user requirements, code comments, generated
explanations or examples after generation. Such post-processing cannot refund
tokens already generated. The output cap remains sufficient for full functions;
reaching it is reported, returns exit code2 and is never labeled verified success.

Auto mode uses a shorter profile only if it passes the measured quality/cost
gates for the same model revision, source, device and dtype. Otherwise it retains
the previous concise prompt. Explicit `--profile legacy|lean|checked` is available
for experiments. No hidden retries, gold-answer routing, donor pass or second
model verifier is used. Default budget is 1024 tokens for every kind (raised from 320 after hard-set evidence); actual
cost follows generated tokens through EOS, not the unused budget.

Reproduce the study in a fresh result directory and a fresh policy path:

```sh
independent_audit/.venv/bin/python -m pytest independent_audit/tests lean_harness/tests -q
independent_audit/.venv/bin/python -u -m lean_harness.benchmark --out lean_harness/results/new_run --policy-out lean_harness/new_policy.json
```

This compares three profiles on16 development tasks, freezes one candidate per
kind, then checks32 fresh validation tasks and37 prior regression tasks. Input
plus output tokens and whole-request wall time are counted. A new default must
save both without item-level correctness or stopping regressions; the prior suite
additionally requires preserving its best observed answers. Small task sets and
finite tests do not certify universal accuracy or production code quality.
