# Fixed-map partition results

Fixed original donor and position map. Disjoint masks, nonadditive effects; empty masks are analytical identities with no measured latency.

| Suite | Partition | Task checks | Checks + EOS | Measured / analytical | Same as B / E | Mean tokens |
|---|---|---:|---:|---:|---:|---:|
| coding | chat_or_boundary | 1/4 | 0/4 | 4 / 0 | 2 / 0 | 320.00 |
| coding | instruction | 1/4 | 0/4 | 4 / 0 | 0 / 0 | 320.00 |
| coding | question | 0/4 | 0/4 | 4 / 0 | 0 / 0 | 320.00 |
| heldout_arithmetic | chat_or_boundary | 8/8 | 8/8 | 8 / 0 | 6 / 0 | 229.25 |
| heldout_arithmetic | instruction | 8/8 | 8/8 | 8 / 0 | 0 / 5 | 161.62 |
| heldout_arithmetic | question | 6/8 | 6/8 | 2 / 6 | 6 / 0 | 221.75 |
| original | chat_or_boundary | 25/25 | 25/25 | 25 / 0 | 19 / 0 | 175.68 |
| original | instruction | 25/25 | 25/25 | 25 / 0 | 0 / 23 | 95.24 |
| original | question | 24/25 | 24/25 | 0 / 25 | 25 / 0 | 175.20 |
