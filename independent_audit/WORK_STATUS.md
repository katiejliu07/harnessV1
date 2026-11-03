# Completed audit status

All requested measurements and final manual reviews are complete. See
[FINAL_REPORT.md](FINAL_REPORT.md) for findings, scope and limitations.

- Upstream commit: `6c54c6926f508bb7515b96c52f54e03eb3949327`.
- Original tracked files are unchanged; the original artifact manifest verifies84
  files. All new work is under independent_audit/.
- Locked environment: framework CPython3.12.3, PyTorch2.7.1, TransformerLens2.16.1,
  Transformers4.57.1, MPS float16 on an18GB M3 Pro. No CUDA results.
- Main matrix:790 records, no errors/skips, all16 conditions on37 tasks plus
  changed-donor probes. All92 no-op/alpha0 comparisons match full token IDs.
- Original-path reproduction:25 questions, baseline24/25, patched25/25.
  Saved historical v8 final-answer rescoring is25/25 versus25/25.
- Full arithmetic/native/cache parity passes25/25 baseline and25/25 patch.
  Actual coding native/full/cache parity passes all4 tasks.
- Exact donor early-stop:37 prompts,3 paired repeats, bitwise-identical captured
  tensors, donor-only median-time ratio0.26758.
- Prefix partitions:111 records,80 measured generations and31 analytical empty
  masks. The partition summary validates disjoint recovery of E's original map.
- Timing repeats:5 tasks x4 methods x3 repeats =60 records; no token disagreements.
- Fixed-length donors:6 targets x4 interventions =24 comparisons, zero answer
  changes, all24 full token sequences change.
- Common1024-budget diagnostic:5 conditionally selected targets x4 methods =20
  records; original token prefixes unchanged. Coding A/C/E4/4, B3/4 after one
  explicit adjudication of repeated identical functions. B's bracket failure is
  real; its counter ignores nesting.
- Historical text audit:954 response records in33 arrays, all retained by source.
- Final full unit suite:33 passed (5.18s). Log:
  results/followup_logs/final_unit_tests.txt.
- Both serial queues ended successfully; no queued GPU work remains.
- Final receipt/table exporter: finalize.py; output results/verification.json and
  conditions.csv alongside each principal summary.

Keep the six inference-source files hashed in the main plan unchanged if
resuming this exact run. New inference semantics require a fresh plan/directory.
Raw scores/data are immutable evidence; later grading corrections and the one
manual adjudication are separately bound by SHA-256 and fully documented in
GRADING_CHANGES.md. Cost proxies are not CUDA FLOPs, energy or dollars. Passing
finite tests on four functions does not certify general code quality.
