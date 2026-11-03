# Token-light prompt benchmark

Small synthetic/task-level tests, not universal code quality. Selection on development only, validation unchanged after selection. No post-generation stripping, oracle retries or extra model verification calls.

| Stage | Kind | Profile | Checks / EOS | Mean input | Mean output | Mean total tokens | Mean request s |
|---|---|---|---:|---:|---:|---:|---:|
| development | code | legacy | 3/4 / 3/4 | 82.25 | 103.50 | 185.75 | 11.013 |
| development | code | nosys | 3/4 / 3/4 | 61.25 | 162.50 | 223.75 | 19.601 |
| development | code | steps | 4/4 / 4/4 | 77.25 | 77.50 | 154.75 | 7.986 |
| development | code | terse | 4/4 / 4/4 | 77.25 | 77.50 | 154.75 | 8.474 |
| development | scalar | legacy | 11/12 / 11/12 | 64.25 | 93.58 | 157.83 | 11.724 |
| development | scalar | nosys | 12/12 / 12/12 | 43.25 | 104.92 | 148.17 | 11.047 |
| development | scalar | steps | 11/12 / 11/12 | 58.25 | 38.42 | 96.67 | 3.893 |
| development | scalar | terse | 9/12 / 9/12 | 46.25 | 37.58 | 83.83 | 4.021 |
| regression | code | legacy | 4/4 / 4/4 | 77.50 | 111.50 | 189.00 | 12.021 |
| regression | code | steps | 3/4 / 3/4 | 72.50 | 63.50 | 136.00 | 6.302 |
| regression | scalar | legacy | 32/33 / 32/33 | 61.24 | 55.94 | 117.18 | 5.636 |
| regression | scalar | nosys | 32/33 / 32/33 | 40.24 | 56.03 | 96.27 | 5.525 |
| validation | code | legacy | 6/8 / 6/8 | 78.38 | 109.12 | 187.50 | 11.717 |
| validation | code | steps | 5/8 / 5/8 | 73.38 | 73.25 | 146.62 | 8.055 |
| validation | scalar | legacy | 20/24 / 20/24 | 64.42 | 91.75 | 156.17 | 9.770 |
| validation | scalar | nosys | 23/24 / 23/24 | 43.42 | 98.58 | 142.00 | 10.419 |

Policy decisions:

- scalar: selected nosys; promoted=False; old best results preserved=False.
- code: selected steps; promoted=False; old best results preserved=False.
