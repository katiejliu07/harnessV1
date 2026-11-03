# Independent activation-patching audit

Audited upstream commit: `6c54c6926f508bb7515b96c52f54e03eb3949327`.
All original source/results remain unchanged. This directory contains new code
and separately named evidence, including unsuccessful backend checks.

Completed results and interpretation: [FINAL_REPORT.md](FINAL_REPORT.md).
The main matrix has 790 records, all serial followups are complete, and the final
suite passes 33 tests. Original tracked files remain unchanged. The integrity
receipt is `results/verification.json`; CSV tables accompany the main, repeated-
timing and common-budget summaries.

## Reproduction

Use framework CPython 3.12.3 on this Mac, not its Anaconda installation (readline
segfaulted during pytest startup). `uv.lock` pins all direct/transitive packages
and distribution hashes; it does not include model weights. From this directory:

```sh
uv sync --frozen --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

From repository root, with the resulting interpreter:

```sh
independent_audit/.venv/bin/python -m independent_audit.model
independent_audit/.venv/bin/python -m independent_audit.saved_audit
independent_audit/.venv/bin/python -m independent_audit.validate_artifacts
independent_audit/.venv/bin/python -m independent_audit.historical_text_audit
independent_audit/.venv/bin/python -m independent_audit.auxiliary_audit
independent_audit/.venv/bin/python -m pytest independent_audit/tests -q
independent_audit/.venv/bin/python -u -m independent_audit.benchmark --device mps --dtype float16 --out independent_audit/results/mps_torch271_full --reproduce --extensions
independent_audit/.venv/bin/python -m independent_audit.quality_review independent_audit/results/mps_torch271_full
independent_audit/.venv/bin/python -m independent_audit.report independent_audit/results/mps_torch271_full
independent_audit/.venv/bin/python -m independent_audit.optimization --device mps --dtype float16
```

The saved main run is complete. Reusing its directory resumes saved records; it
does not produce a fresh timing sample. Use a new `--out` for a fresh run.
Only `completed.json` confirms all requested rows were generated. `plan.json` is
written before model loading and cannot silently change on resume. Every record
retains input/output IDs, donor question and map, correctness, truncation,
donor-forward/generation/total timings, and memory observations. Use a new output
directory after changes to task/prompt/model/engine semantics. Partial files are
never presented as a completed benchmark.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its snapshot is in `.audit-model/`.
The original model revision and original dependency versions were not recorded;
historical bitwise replication is not identifiable from the repository alone.

## Findings established from original code and data

* Saved v8 final-answer accuracy is **25/25 versus 25/25**, not 23/25 versus 25/25.
  Correct the extraction without rewriting the original JSON. The response for
  81/9 nonetheless includes a false intermediate statement (90 is less than 81),
  so final-answer correctness does not validate reasoning quality.
* Stored token counts give means 173.56 and 94.88, difference 78.68. Paired t is
  10.217975, p = 3.220672e-10, dz = 2.043595. The README's p < 1e-10 does not
  match the included data. A fixed-seed, 10,000-resample paired percentile
  interval is [64.399, 93.84] tokens. This is descriptive after model selection.
* The logs contain 51 search records, including 24 quick and 11 full-stage
  records, and 37 distinct serialized configurations. These are not independent
  test sets or independent confirmations of the selected result.
* v8 baseline uses native cached generation; its intervention uses the growing
  full sequence without KV reuse. Token reduction cannot establish time savings.
* Absolute-prefix copying maps a donor's question/chat tokens into a different
  target instruction context. The new benchmark records actual token maps and
  separately patches question spans, instruction spans, and identical suffixes.
  The exact tokenizer audit finds zero question tokens touched in all 25
  original prompts, but question tokens are touched in two of eight arithmetic
  extensions and all four coding prompts (12--30 tokens in the latter). The
  intervention's semantic scope therefore changes with prompt length. Full CSV
  maps are in `results/alignment_maps/`.
* The original EFFECTIVE test accepts 18/25 maintained correct answers even if
  baseline is 25/25: seven lost answers (28%). The independent observed-sample
  gate requires zero regressions and lower measured total cost, and makes no
  population noninferiority claim.
* `generation_phase_analysis.py` attaches a prefix hook inside native cached
  generation. Under the pinned library's default cache, local position zero is
  overwritten with donor position zero during decode. Its generation-phase
  intervention therefore differs from the v8 prompt-only full-recompute method.
  All 25 saved intervened `output_preview` values in that probe are whitespace.
  The README's final-layer rank change (74970.2 to 131.88) comes from this
  degenerate probe, not a successful v8 answer sequence. Its single-token
  answer-position search also misses answers split across multiple tokens.
  Its offline full-sequence replay patches prompt positions instead of the
  generated positions overwritten online, so that replay is not a faithful
  trace of the intervention that produced the output.
* Prompt-end cosine similarity cannot isolate style; logit-lens scoring takes
  only the first token of a gold answer and is not whole-answer correctness.
* `setup_validation.py` changes both output budgets and sampling settings between
  normal and reasoning conditions, and changes them again on CPU. Its token
  ratios are not a controlled prompt-only comparison. The single-layer script
  attempts an unsupported `fwd_hooks` generation argument in the pinned library
  and falls back to full recomputation; behavior depends on the library version.
* `compute_quality_stats.py` selects the first strategy from a multi-strategy
  file. The independent audit traverses every detailed-result record instead.
  Auxiliary validation parses all 22 original JSON files and replays 208 saved
  layer/head token-change fields; all these arithmetic deltas are consistent.
  Count-only artifacts do not contain enough information to recover correctness.

## Controls and quality

A normal question; B original reasoning prompt; C concise prompt; C exact-length
concise control where a rendered-token-length match exists; D no-op layers;
E original [6,2,3] residual-post prefix; F layers 2/3/6 individually; G fixed
random layer, shuffled positions, unrelated donor, alpha zero; H instruction,
question, and identical suffix. E is also the requested original-prefix H
control; it is not counted as an independent duplicate experiment.

Main conditions share greedy argmax, chat template, batch size one, tokenizer,
dtype, 320-token cap and EOS. The native API, no-op, alpha-zero, original-hook
and recomputed paths undergo parity checks first. This is prompt-only caching:
time-limited interventions are rejected because cached patched states would need
invalidation when removing the patch.

Frozen extensions contain eight arithmetic tasks and four Python tasks, with
executable tests including edge cases and input immutability. Four tasks cannot
validate general code quality. The restricted evaluator rejects imports and
unsafe/unsupported syntax, has a child-process timeout and CPU limit, and is not
a general untrusted-code sandbox. Full output text and errors remain available.

`quality_review.py` and `code_quality.py` add separately recorded diagnostics
without overwriting the frozen raw scores. The supplemental grader accepts safe
standard-library solutions when imports were not prohibited, excludes example
calls from function tests, and tests the required binary-search access bound at
three sizes. Its tests were written before the coding generations began. Answer
format adherence is reported separately from scalar correctness. These checks
still do not establish the soundness of every generated explanation.
The supplemental scalar scorer is separately versioned as `scoring_v2.py`: a
review caught a frozen-parser false negative when "our answer is correct"
preceded a verification equation and the real conclusion. Raw outputs/scores
remain unchanged; supplemental corrected scores and raw counts both appear in
the final summary. The two original saved-v8 extraction corrections are preserved.

Donor probes fix each target and change to two different-answer questions for
every patched control on the first three targets per suite. Record changes and
donor-answer attraction separately; logits changing alone does not prove leakage,
and a handful of negative probes cannot establish absence of leakage.

## Cost and backend limitations

End-to-end per-task time includes tokenization, position alignment, donor capture,
hooked prefill/decode and output decoding; executable grading is timed separately
from inference. Model load is recorded separately and can be added for cold use.
Donors are not silently amortized. Model-only seconds and forward-token/attention
pair proxies are also saved. Proxies are not FLOPs, energy, or money. MPS current
allocation is not a peak; process RSS is a lifetime maximum, not a per-task peak.
CUDA peak allocation is only meaningful when CUDA is actually available.

Generation TTFT and query/attention work proxies exclude donor preparation;
donor input length and time are recorded separately. The frozen CUDA allocation
counter is not reset per request, so it is a running peak, not an isolated task
peak. `ru_maxrss` is bytes on this Mac; a Linux run would require conversion from
KiB before treating its raw RSS field as bytes. These memory observations are not
used to assert a memory saving.

The local M3 Pro has 18 GB unified memory and no CUDA. The first half-precision
Metal pilot produced nonsensical text even though generation paths agreed; those
outputs are invalid for method evaluation. A full-precision Metal pilot was
interrupted after memory pressure/stalling. Ordinary CPU float32 Qwen passed a
direct Hugging Face sanity check. See run-specific environment and completion
files for the actual valid benchmark status. Do not extrapolate CPU timings to
RTX 4070, or treat this machine's shared interactive load as a controlled GPU lab.

The initial environment used PyTorch 2.8.0. A subsequently located upstream
[TransformerLens issue](https://github.com/TransformerLensOrg/TransformerLens/issues/1008)
reports a PyTorch 2.8 MPS regression affecting noncontiguous tensors and says
2.7.1 restored valid Qwen output. The current lock pins 2.7.1; semantic sanity,
full arithmetic/coding path parity and all no-op/alpha0 controls passed in the
completed run. `failed_environment/` preserves the former lock and specification.

## Followups and rescoring

Run GPU commands serially, matching the main run's device and dtype. Each
measurement helper accepts a fresh output path; the partition, coding-parity,
fixed-length donor and budget helpers also accept `--main` for a fresh main
directory. Coding parity uses `--out-stem` rather than `--out`. Use new names
when repeating a completed measurement so prior evidence remains available.

```sh
independent_audit/.venv/bin/python -u -m independent_audit.prefix_partitions
independent_audit/.venv/bin/python -m independent_audit.partition_report
independent_audit/.venv/bin/python -u -m independent_audit.benchmark --device mps --dtype float16 --out independent_audit/results/mps_torch271_timing_repeats --limit 5 --donor-probes 0 --repeats 3 --conditions A_normal B_reasoning C_concise E_v8_prefix
independent_audit/.venv/bin/python -u -m independent_audit.code_cache_parity
independent_audit/.venv/bin/python -u -m independent_audit.equal_length_donors
independent_audit/.venv/bin/python -u -m independent_audit.budget_followup
independent_audit/.venv/bin/python -m independent_audit.quality_review independent_audit/results/budget_1024_followup
independent_audit/.venv/bin/python -m independent_audit.adjudicate_repeated_solution
independent_audit/.venv/bin/python -m independent_audit.report independent_audit/results/budget_1024_followup
```

Apply `quality_review` then `report` to every new benchmark directory, including
the timing directory. Explicit manual adjudications are bound to response/data
hashes and retain automatic results. The repeated-solution review only accepts
two AST-identical passing definitions, never the best of conflicting answers.
See [GRADING_CHANGES.md](GRADING_CHANGES.md) for the full audit trail.

`postprocess.py` and `finish_measurements.py` are the serial queues used for this
particular saved run; their stage files preserve completion provenance. They
have fixed run-directory conventions. For an alternate backend or fresh directory,
invoke the individual CLI helpers above with matching parameters instead.
`finalize.py` checks this saved run's integrity/coverage and exports its tables;
it is not a substitute for running tests or interpreting the measurements.
