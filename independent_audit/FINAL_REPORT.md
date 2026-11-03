# Independent activation-patching audit

Completed: the 790-record main benchmark, original-path reproduction, causal
controls, timing repeats, coding parity, fixed-length donor tests and common-budget
diagnostic. The final suite passes **33 tests**. New results preserve original
files and distinguish functional success, output stopping and measured cost.

## Scope and reproducibility

Upstream commit: `6c54c6926f508bb7515b96c52f54e03eb3949327`. Original
experiment source and results were preserved. All new work is under
`independent_audit/`. The original-artifact manifest verifies 84 files; tracked
Git changes are checked separately.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The safetensors SHA-256 matches
the published Hugging Face LFS object. The upstream experiment did not record
its model revision or dependency versions, so historical bitwise reproduction
cannot be guaranteed.

Hardware: Apple M3 Pro, 18 GPU cores, 18 GB unified memory, macOS 15.1. CUDA is
unavailable. Measurements use MPS float16, CPython 3.12.3, PyTorch 2.7.1,
TransformerLens 2.16.1 and Transformers 4.57.1. `uv.lock` pins transitive
dependencies and package hashes. Full package/configuration records are in the
run's `environment.json`; hardware and model-integrity records are in `results/`.

The first PyTorch 2.8 MPS pilot produced invalid ordinary outputs despite path
parity. Failed artifacts and its lock are retained, explicitly excluded from
method evidence. A documented [upstream MPS regression](https://github.com/TransformerLensOrg/TransformerLens/issues/1008)
suggested the 2.7.1 retry; semantic sanity and parity were then independently
verified. CPU float32 Hugging Face inference also passed a direct sanity check.
These measurements do not validate CUDA performance or all MPS configurations.

## Saved-data audit and original-path reproduction

The saved v8 data give **25/25 correct final answers for both baseline and patch**,
not 23/25 versus 25/25. Original extraction mistakes affected 81/9 and 14% of 200.
No original result file was rewritten. Final-answer accuracy does not validate
reasoning: the saved 81/9 explanation contains the false claim that 90 is less
than 81.

Saved output-token means are 173.56 and 94.88: a paired reduction of 78.68 tokens.
Recalculated paired t = 10.217975, p = 3.220672e-10, and dz = 2.043595. The
README's p < 1e-10 does not match these data. A fixed-seed 10,000-resample paired
percentile interval for the mean reduction is [64.399, 93.84] tokens. These are
descriptive statistics after selection, not independent confirmatory inference.

All 50 saved response texts retokenize to one token fewer than their stored
counts, consistent with a discarded EOS. Historical token IDs were not saved,
so this cannot establish their exact original tokenization. An identically
copied quality-assessment JSON is not a second replication. Auxiliary checks
parsed all 22 original JSON files and reproduced 208 stored layer/head token
deltas. Count-only artifacts cannot supply missing correctness evidence.

The exhaustive text audit covers 954 response records in 33 quality arrays,
including duplicate-file provenance. Stored minus retokenized counts are zero
for 139, one for 814 and two for one response. Historical whitespace/EOS and
tokenizer differences prevent treating these as automatic count errors.
The V2 parser changes one additional earlier v4 scalar score: the response gives
`x = 14/2`, equivalent to 7, then cuts off before finishing the explanation.
That is a correct intermediate solution value, not a complete final-answer
response. Both scoring versions and the response reference are retained.

The fresh original-v8-path run gave baseline **24/25** and patch **25/25**. Its
baseline genuinely answered 7 times 8 as 35. Patched text matched the historical
saved text on 25/25 questions; baseline text matched on 21/25. Fresh token means
were 175.20 and 94.88.

Critically, the original generation paths averaged **14.327 s baseline versus
29.967 s patched**. These legacy timings include generation and, for patching,
the donor forward, but exclude tokenization and output decoding. The shorter
patched responses therefore cost more inference time in the original code.

## Fair main benchmark

The frozen matrix contains 37 tasks: the reused 25 arithmetic questions, eight
new arithmetic questions, and four small Python tasks. Sixteen conditions per
task give 592 records; 198 changed-donor probes give 790 total. All records
completed without a generation error or skipped condition. Repeated donors and
controls are not independent accuracy trials.

Conditions A/B/C/D/E, single layers 2/3/6, a fixed random layer, shuffled donor
positions, unrelated donors, alpha=0, instruction-only alignment, question-only
alignment, shared suffix and exact-length concise controls share the tokenizer,
chat template, greedy decoding, EOS, batch size one, dtype, execution backend,
and 320-token cap. E also supplies the original-prefix positional control.
The exact-length concise control pads with `Please`; such padding is recorded
and is not assumed semantically neutral. Programming uses an explicitly adapted
step-by-step prompt requesting a Python function, rather than the numeric
final-answer template.

Each primary time below includes tokenization/alignment, donor preparation when
needed, generation and output decoding. Model loading and executable grading
are separately recorded and excluded. No donor cost is silently amortized.

| Suite | Condition | Task checks passed | Checks plus EOS before cap | Mean tokens | Mean request s |
|---|---|---:|---:|---:|---:|
| Original arithmetic, n=25 | A normal | 25/25 | 25/25 | 61.52 | 4.855 |
| | B reasoning | 24/25 | 24/25 | 175.20 | 13.058 |
| | C concise | 25/25 | 25/25 | 40.76 | 3.208 |
| | E patched | 25/25 | 25/25 | 94.88 | 7.191 |
| New arithmetic, n=8 | A normal | 8/8 | 7/8 | 142.88 | 10.893 |
| | B reasoning | 8/8 | 8/8 | 231.00 | 18.114 |
| | C concise | 7/8 | 7/8 | 103.38 | 8.319 |
| | E patched | 8/8 | 8/8 | 155.12 | 11.757 |
| Python, n=4 | A normal | 4/4 | 0/4 | 320.00 | 24.585 |
| | B reasoning | 1/4 | 0/4 | 320.00 | 24.200 |
| | C concise | 4/4 | 4/4 | 111.50 | 8.420 |
| | E patched | 3/4 | 0/4 | 320.00 | 25.766 |

All sixteen conditions, paired descriptive intervals, individual regression
indices and donor probes appear in
[the main summary](results/mps_torch271_full/SUMMARY.md) and its adjacent JSON.

EOS is an operational stopping rule, not proof of semantic completeness. A
budget-limited answer may already contain the correct number or a complete
working function. For example, normal prompting gives the correct 97.2-dollar
answer before reaching its cap; all four normal coding outputs contain passing
functions even though their later material reaches the cap. E's binary-search
function itself is unfinished at 320. The concise arithmetic answer 390 for
13/16 of 240 is a genuine error; the correct answer is 195.

Coding checks enforce return types, empty inputs, representative edge cases,
input immutability and, for binary search, 15 counted-access probes at sizes
256, 4096 and 65536. All four concise functions were also inspected. Finite
tests and access counts do not prove general correctness, asymptotic complexity,
maintainability or explanation soundness. Supplemental grading accepts permitted
safe standard-library imports and complete Python inside an unclosed Markdown
fence; it never repairs incomplete Python. Raw grades remain unchanged, and
input/diagnostic/grader hashes bind each supplemental review. See
[the grading audit trail](GRADING_CHANGES.md).

The observed-sample cost/quality guard separately checks task-check regressions
and checks-plus-EOS regressions. Two budget-limited responses must not conceal a
loss of working code. It also rejects batches with no checks-plus-EOS successes.
Passing this guard does not establish population noninferiority or certify all
quality dimensions.

## Implementation optimization and parity

The new generator patches prompt residuals once, caches their consequences, and
then decodes with KV reuse. It never copies donor position zero into decode
position zero. Removing a time-limited patch would require KV invalidation;
unsupported schedules are rejected instead of silently changing the experiment.

All 92 full-run no-op/alpha-zero comparisons reproduce baseline token IDs exactly.
Prompt logits with an identity hook have maximum absolute difference zero.
Original native baseline versus new cached baseline matches on all 25 complete
arithmetic generations; original full-recompute patch versus cached patch also
matches on all 25. These are token-output checks: FP16 patched logits can differ
between cached/full paths (warmup maximum difference 0.03125), so no universal
bitwise-logit guarantee is asserted.

The cached/selective-capture implementation reduces the arithmetic patch's
generation-plus-donor time by **4.17 times** with identical tested
outputs. Compare model-phase times to the legacy model-phase times, rather than
mixing them with full request latency. On the original arithmetic matrix,
concise prompting is still cheaper than cached patching. On the new arithmetic
set, patching is cheaper than B but slower than A; concise prompting loses one
correct answer. There is no single demonstrated winner across quality and cost.

An additional exact optimization stops donor execution after block 6 instead of
running all 28 blocks. Across 37 prompts and three alternating-order repetitions,
captured layer-2/3/6 tensors were bitwise identical. The ratio of summed per-task
median donor times was 0.26758 (about 3.74 times faster). This is a donor-only
measurement, not a measured 3.74-times full-request improvement. The frozen main
matrix retains its original donor-capture implementation.

## Causal and statistical limits

Absolute-prefix copying has different semantic scope as prompts change. Exact
token maps touch no target question tokens in the original 25 tasks, but touch
question tokens in two of eight new arithmetic tasks and all four programming
tasks (12–30 question tokens). Donor question states also map into target
instruction positions. Replacing activations is not the same as editing token
text, but it can transplant task information and override instructions.

The layer-2-only patch reproduces E's entire token output on 31 of 33 arithmetic
tasks. This ablation does not establish that all three selected layers are
necessary. Main changed-donor probes contain task transfer: E can answer a car
speed donor with 48 instead of the target percentage question's 63. Another E
probe first solves a donor algebra task before truncating on the target. Those
particular donor changes also change prefix coverage, so they alone do not
separate content sensitivity from patch extent.

The exact fixed-map decomposition contains 80 measured generations and 31
analytical empty-mask identities. Instruction positions alone reproduce E's
entire output on 23/25 original questions and 5/8 new arithmetic questions;
the original-set mean is 95.24 tokens versus E's 94.88. The original question
partition is empty on 25/25 prompts, so it simply preserves B. On the new set,
both nonempty question partitions fail their scalar checks, while the six empty
ones preserve B. Coding instruction/question/chat partitions pass function
checks on 1/4, 0/4 and 1/4 tasks, respectively, within the same 320-token budget.
Their effects are nonadditive and do not supply a unique mechanism. Full results
and mask-coverage checks are in the [partition summary](results/prefix_partition_followup/SUMMARY.md).

Across all main scalar donor probes, 29/132 extracted answers change and 13/132
match the donor answer. Coding functional pass status changes in 22/66 probes;
the donor function appears without the target function in 31/66. These counts
pool distinct interventions, including the question-only transfer control; they
must not all be attributed to main E. Answer changes alone are not necessarily
wrong answers, and absent attraction in a few probes cannot establish no leakage.

The original generation-phase probe is not a valid mechanistic explanation of
successful v8 behavior. Its cached hook repatches local decode position zero,
unlike v8's prompt-only intervention. All 25 saved intervened previews are
whitespace. Its offline replay applies a different patch schedule, and its
single-token digit matcher misses multi-token answers. The README's rank change
from 74970.2 to 131.88 comes from this degenerate probe. First-answer-token
logit-lens scores and prompt-end cosine similarity do not establish whole-answer
correctness or a style-specific causal mechanism.

The repository contains 51 search records and 37 distinct serialized
configurations; repeated use of small problem sets limits inference after
selection. The original EFFECTIVE classification can accept 18/25 maintained
correct answers from a 25/25 baseline, allowing seven lost answers. Other scripts
also change sampling/budgets, mix generation paths, or select only the first
strategy in a multi-strategy file. Regression tests and independently named
rescoring artifacts document these issues without rewriting history.

Wall-clock measurements are local observations on a shared interactive MPS host.
Output tokens and dense attention/query-token counters are not energy, dollars
or hardware-independent FLOPs. Generation work proxies exclude donor execution,
whose input length and time are separately recorded. MPS current allocation is
not peak memory; process RSS is a lifetime peak. No CUDA peak, energy or GPU-dollar
claim is supported here. Eight arithmetic extensions and four toy coding tasks
cannot certify production code quality or an isolated reasoning-style mechanism.

## Followup measurements

Full-length coding checks reproduce all four patched token sequences with the
original full-recompute and new cached generators. All four native unpatched
sequences also match the matrix. The ratio of summed generation times is 2.44
(full/cached), using the same precomputed donor. These times exclude donor
preparation and are not full-request costs. The outputs and executable checks
are retained in `results/code_cache_parity.json`.

The five-task, three-repeat timing check has no repeat token disagreements.
Mean per-task median full-request times are A=2.632 s, B=10.075 s, C=1.277 s and
E=5.064 s; all five tasks pass under each method. These 60 observations are
timing repeats of five existing questions, not 60 new accuracy trials. Runtime
also varies between measurement phases on this shared host; a cached 320-token
coding run in the parity phase took 32.6–53.8 s versus roughly 23–30 s in the
main matrix. Cross-phase timings must not be treated as stable hardware constants.

Fixed-length donor tests change only numerals on six predefined targets, keeping
target IDs, donor lengths and absolute patch maps fixed. E and each single-layer
ablation retain correct target answers on all six: **zero answer changes in 24
comparisons**. Every full token sequence changes. This is negative evidence for
answer leakage in these particular numeral substitutions, not proof of no
content sensitivity. The broader main matrix also contains a same-length,
same-map layer-6 probe where changing the donor task to 6 times 4 makes the
7-plus-5 target answer 11. That is single-layer content sensitivity, not main-E
leakage or direct copying of the donor answer 24. Main E's reported task-transfer
failures use broader changes that also alter patch coverage.

The symmetric 1024-token diagnostic reruns A/B/C/E on every target where any of
these methods reached the original cap: four coding tasks and the discount/tax
question, totaling 20 responses. Every response preserves its original token
prefix, and all terminate before 1024. This conditional subset is not a new
blind evaluation or a replacement for the 320-token matrix.

| Coding at 1024, n=4 | Function checks and EOS | Mean tokens | Mean request s |
|---|---:|---:|---:|
| A normal | 4/4 | 465.00 | 33.953 |
| B reasoning, adjudicated | 3/4 | 590.25 | 52.616 |
| C concise | 4/4 | 111.50 | 8.077 |
| E patched | 4/4 | 470.25 | 34.576 |

B's dedup response repeats two AST-identical, passing functions. The automatic
grader flags multiple solutions as unsupported; a hash-bound explicit review
tests both definitions and corrects this case without selecting a favorable
version. The automatic count2/4 and raw grades remain in JSON. B's remaining
failure is genuine: its counter-based bracket function accepts `([)]`, which is
not correctly nested. E's previously unfinished binary search completes and
passes at the larger budget. All methods give the correct discount/tax answer.
The [budget summary](results/budget_1024_followup/SUMMARY.md) includes its actual
per-condition times and the adjudication reference.

## Recommendation and deliverables

Use the verified cached prompt-only implementation when reproducing this
intervention, with explicit position maps and parity checks. Early-stop donor
capture is an additional independently verified optimization. The evidence
supports these implementation improvements on the measured model and backend.

The quality/cost case for activation patching remains conditional. It reduces
cost against the deliberately verbose arithmetic prompt, but normal/concise
prompt controls are competitive or cheaper. Concise prompting makes one genuine
error on the arithmetic extensions; the four-function coding result is too small
to justify a universal policy. A sufficiently large output budget and executable
validation matter. The controls support instruction and task-information effects;
they do not isolate an intrinsic reasoning-style mechanism.

All raw generations, failed backend pilots, frozen plans, source/model revisions,
supplemental grader manifests, explicit adjudication and paired statistics remain
available. `results/verification.json` binds the final artifacts and source hashes,
checks original preservation and measurement coverage, and accompanies CSV tables
next to each principal summary. The final test log is
`results/followup_logs/final_unit_tests.txt`. Reproduction commands are in
[README.md](README.md).
