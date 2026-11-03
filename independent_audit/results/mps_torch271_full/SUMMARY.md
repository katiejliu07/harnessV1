# Independent benchmark results

Source: `independent_audit/results/mps_torch271_full`. Task-specific costs include donor forwards; output tokens include EOS.

Supplemental executable/complexity grading applied to 130 code rows; raw primary scores remain available in JSON. The success column requires passed task checks AND EOS before the cap. JSON separately reports checks passed even when the budget is reached. EOS is an operational stopping criterion, not proof of semantic completeness. Scalar scoring does not establish reasoning soundness.

| Suite | Condition | Task checks passed | Checks passed & EOS | Budget reached | Mean output tokens | Mean task median total s |
|---|---|---:|---:|---:|---:|---:|
| coding | A_normal | 4/4 | 0/4 | 4 | 320.00 | 24.585 |
| coding | B_reasoning | 1/4 | 0/4 | 4 | 320.00 | 24.200 |
| coding | C_concise | 4/4 | 4/4 | 0 | 111.50 | 8.420 |
| coding | C_length_matched | 3/4 | 3/4 | 0 | 91.50 | 7.080 |
| coding | D_noop | 1/4 | 0/4 | 4 | 320.00 | 23.893 |
| coding | E_v8_prefix | 3/4 | 0/4 | 4 | 320.00 | 25.766 |
| coding | F_layer2 | 3/4 | 0/4 | 4 | 320.00 | 24.667 |
| coding | F_layer3 | 3/4 | 0/4 | 4 | 320.00 | 24.599 |
| coding | F_layer6 | 3/4 | 0/4 | 4 | 320.00 | 23.683 |
| coding | G_alpha0 | 1/4 | 0/4 | 4 | 320.00 | 24.200 |
| coding | G_random_layer | 1/4 | 0/4 | 4 | 320.00 | 24.195 |
| coding | G_shuffled | 0/4 | 0/4 | 4 | 320.00 | 23.315 |
| coding | G_unrelated | 0/4 | 0/4 | 4 | 320.00 | 24.887 |
| coding | H_instruction | 2/4 | 0/4 | 3 | 317.25 | 23.935 |
| coding | H_question | 1/4 | 0/4 | 4 | 320.00 | 23.945 |
| coding | H_suffix | 2/4 | 0/4 | 4 | 320.00 | 24.028 |
| heldout_arithmetic | A_normal | 8/8 | 7/8 | 1 | 142.88 | 10.893 |
| heldout_arithmetic | B_reasoning | 8/8 | 8/8 | 0 | 231.00 | 18.114 |
| heldout_arithmetic | C_concise | 7/8 | 7/8 | 0 | 103.38 | 8.319 |
| heldout_arithmetic | C_length_matched | 6/8 | 6/8 | 0 | 75.12 | 5.561 |
| heldout_arithmetic | D_noop | 8/8 | 8/8 | 0 | 231.00 | 18.034 |
| heldout_arithmetic | E_v8_prefix | 8/8 | 8/8 | 0 | 155.12 | 11.757 |
| heldout_arithmetic | F_layer2 | 8/8 | 8/8 | 0 | 155.12 | 12.435 |
| heldout_arithmetic | F_layer3 | 8/8 | 8/8 | 0 | 163.25 | 12.528 |
| heldout_arithmetic | F_layer6 | 8/8 | 8/8 | 0 | 165.00 | 12.643 |
| heldout_arithmetic | G_alpha0 | 8/8 | 8/8 | 0 | 231.00 | 17.720 |
| heldout_arithmetic | G_random_layer | 8/8 | 8/8 | 0 | 226.12 | 17.973 |
| heldout_arithmetic | G_shuffled | 7/8 | 7/8 | 0 | 146.12 | 11.177 |
| heldout_arithmetic | G_unrelated | 6/8 | 6/8 | 0 | 169.12 | 12.846 |
| heldout_arithmetic | H_instruction | 8/8 | 8/8 | 0 | 222.88 | 17.603 |
| heldout_arithmetic | H_question | 7/8 | 7/8 | 1 | 243.12 | 18.954 |
| heldout_arithmetic | H_suffix | 8/8 | 7/8 | 1 | 228.88 | 18.286 |
| original | A_normal | 25/25 | 25/25 | 0 | 61.52 | 4.855 |
| original | B_reasoning | 24/25 | 24/25 | 0 | 175.20 | 13.058 |
| original | C_concise | 25/25 | 25/25 | 0 | 40.76 | 3.208 |
| original | C_length_matched | 25/25 | 25/25 | 0 | 48.36 | 4.026 |
| original | D_noop | 24/25 | 24/25 | 0 | 175.20 | 13.185 |
| original | E_v8_prefix | 25/25 | 25/25 | 0 | 94.88 | 7.191 |
| original | F_layer2 | 25/25 | 25/25 | 0 | 98.20 | 7.420 |
| original | F_layer3 | 24/25 | 24/25 | 0 | 97.92 | 7.757 |
| original | F_layer6 | 25/25 | 25/25 | 0 | 104.20 | 7.930 |
| original | G_alpha0 | 24/25 | 24/25 | 0 | 175.20 | 13.218 |
| original | G_random_layer | 24/25 | 24/25 | 0 | 171.12 | 13.015 |
| original | G_shuffled | 24/25 | 24/25 | 0 | 98.76 | 7.868 |
| original | G_unrelated | 24/25 | 24/25 | 0 | 121.80 | 9.527 |
| original | H_instruction | 24/25 | 24/25 | 1 | 178.68 | 13.853 |
| original | H_question | 25/25 | 25/25 | 0 | 182.40 | 14.005 |
| original | H_suffix | 25/25 | 25/25 | 0 | 174.80 | 13.338 |

Scalar donor probes: 132; answers changed in 29; donor-answer attraction in 13.
Coding donor probes: 66; functional pass status changed in 22; donor function name appeared without the target name in 31.

These are observed sample results, not evidence of a style-specific mechanism or broad preservation of code quality.

## Original v8 generation paths

25 questions: baseline 24 correct, patched 25 correct.
Mean token reduction: 80.32. Mean generation-plus-donor time reduction: -15.639 s (negative means patching is slower). Tokenization/decoding are excluded from these legacy timings.

Full-sequence native/manual/cached comparisons are retained in summary.json; short warmup parity alone does not establish long-generation parity.
