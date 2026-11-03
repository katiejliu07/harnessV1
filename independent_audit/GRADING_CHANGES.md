# Supplemental grading audit trail

Inference sources, task cases, prompts, configurations and raw scores are frozen
in the main plan. Grading corrections never edit `benchmark.jsonl`.

The independent scalar parser initially misread a verification phrase, "our
answer is correct", as an answer anchor. `scoring_v2.py` corrects that failure
and retains the first parser's score in the report. Its regression tests include
wrong answers; it never searches for the expected answer in the response.

The supplemental coding harness was initially authored before coding generation.
It accepts permitted safe standard-library imports and excludes example calls,
while enforcing the requested return types, input immutability and binary-search
access bounds. The original restricted raw grader remains preserved.

During review of the third coding task, A_normal had a complete lower_bound
function but reached the output cap in its examples, leaving a Markdown fence
unclosed. The supplemental parser now accepts an unclosed fence when the Python
body itself parses. It does not repair incomplete Python expressions. A regression
test covers both situations. This changes function-test diagnostics, while the
independently recorded response-truncation flag and complete-response score remain
unchanged. Functional tests and expected answers were not changed.

`grading_manifest.json` binds raw data, supplemental diagnostics and grader source
hashes. `report.py` rejects stale grades rather than silently mixing versions.
These finite checks do not certify all aspects of program or explanation quality.

Reporting also guards task checks separately from the joint checks-plus-EOS
indicator. At 320 tokens, normal coding outputs can contain passing functions
even though their later examples reach the cap. Comparing only the joint flag
would conceal functional regressions between two capped responses. The report
now rejects such regressions explicitly, reports both dimensions in its tables,
and requires at least one checks-plus-EOS success before flagging a condition as
eligible on the observed sample. A focused regression test covers this case.

The exhaustive historical-text check found one additional V1/V2 score difference:
`quality_assessment_v4.json`, `root.strategies[21].detailed_results[4].intervened`.
Manual inspection confirms it gives `x = 14/2`, equivalent to the expected 7,
but stops at an open display-math delimiter before completing its explanation.
V2 recognizes the fraction; the first parser took its denominator as the answer.
The V2 scalar success must not be described as a complete final-answer response.
No original or first-audit record was overwritten. Main saved-v8 scores remain
25/25 versus 25/25 under both independent scorers.

The 1024-token B/dedup response offers the same function twice. The automatic
grader deliberately flags multiple solution blocks for review, so it scored
this case unsupported. `adjudicate_repeated_solution.py` verifies that both
offered definitions normalize to identical AST-derived source and independently
pass all fixed tests; it does not select the best of conflicting solutions.
`budget_1024_followup/manual_adjudications.json` records this explicit exception,
bound to the raw data and response SHA-256. Reporting preserves automatic2/4
and raw scores while reporting adjudicated B coding3/4. The remaining B failure
is real: its counter-based bracket checker accepts `([)]`. A reporting test
ensures adjudications cannot silently apply to changed input data and preserves
the automatic verdict. No general grader or frozen inference source changed.
