# Token-light prompt benchmark

Small synthetic/task-level tests, not universal code quality. Selection on development only, validation unchanged after selection. No post-generation stripping, oracle retries or extra model verification calls.

| Stage | Kind | Profile | Checks / EOS | Mean input | Mean output | Mean total tokens | Mean request s |
|---|---|---|---:|---:|---:|---:|---:|
| development | code | legacy | 3/4 / 3/4 | 82.25 | 103.50 | 185.75 | 13.607 |
| development | code | nosys | 3/4 / 3/4 | 61.25 | 162.50 | 223.75 | 23.800 |
| development | code | terse | 4/4 / 4/4 | 77.25 | 77.50 | 154.75 | 24.064 |
| development | scalar | legacy | 11/12 / 11/12 | 64.25 | 93.58 | 157.83 | 10.877 |
| development | scalar | nosys | 12/12 / 12/12 | 43.25 | 104.92 | 148.17 | 12.873 |
| development | scalar | terse | 9/12 / 9/12 | 46.25 | 37.58 | 83.83 | 4.181 |

Policy decisions:

- scalar: selected legacy; promoted=False; old best results preserved=None.
- code: selected legacy; promoted=False; old best results preserved=None.
