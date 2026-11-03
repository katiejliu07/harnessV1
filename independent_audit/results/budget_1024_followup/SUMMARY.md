# Independent benchmark results

Source: `independent_audit/results/budget_1024_followup`. Task-specific costs include donor forwards; output tokens include EOS.

Supplemental executable/complexity grading applied to 16 code rows; raw primary scores remain available in JSON. The success column requires passed task checks AND EOS before the cap. JSON separately reports checks passed even when the budget is reached. EOS is an operational stopping criterion, not proof of semantic completeness. Scalar scoring does not establish reasoning soundness.

| Suite | Condition | Task checks passed | Checks passed & EOS | Budget reached | Mean output tokens | Mean task median total s |
|---|---|---:|---:|---:|---:|---:|
| coding | A_normal | 4/4 | 4/4 | 0 | 465.00 | 33.953 |
| coding | B_reasoning | 3/4 | 3/4 | 0 | 590.25 | 52.616 |
| coding | C_concise | 4/4 | 4/4 | 0 | 111.50 | 8.077 |
| coding | E_v8_prefix | 4/4 | 4/4 | 0 | 470.25 | 34.576 |
| heldout_arithmetic | A_normal | 1/1 | 1/1 | 0 | 323.00 | 24.416 |
| heldout_arithmetic | B_reasoning | 1/1 | 1/1 | 0 | 290.00 | 20.744 |
| heldout_arithmetic | C_concise | 1/1 | 1/1 | 0 | 250.00 | 17.831 |
| heldout_arithmetic | E_v8_prefix | 1/1 | 1/1 | 0 | 210.00 | 14.663 |

Explicit manual adjudications applied to 1 response(s); automatic task-check counts and all evidence remain in summary.json and manual_adjudications.json.

Scalar donor probes: 0; answers changed in 0; donor-answer attraction in 0.
Coding donor probes: 0; functional pass status changed in 0; donor function name appeared without the target name in 0.

These are observed sample results, not evidence of a style-specific mechanism or broad preservation of code quality.
