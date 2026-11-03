# Token-light prompt benchmark

Small synthetic/task-level tests, not universal code quality. Selection on development only, validation unchanged after selection. No post-generation stripping, oracle retries or extra model verification calls.

| Stage | Kind | Profile | Checks / EOS | Mean input | Mean output | Mean total tokens | Mean request s |
|---|---|---|---:|---:|---:|---:|---:|
| development | code | compact | 4/4 / 4/4 | 76.25 | 196.50 | 272.75 | 22.075 |
| development | code | legacy | 3/4 / 3/4 | 82.25 | 103.50 | 185.75 | 10.848 |
| development | scalar | compact | 8/12 / 8/12 | 55.33 | 29.08 | 84.42 | 3.058 |
| development | scalar | legacy | 11/12 / 11/12 | 64.25 | 93.58 | 157.83 | 11.406 |

Policy decisions:

- scalar: selected legacy; promoted=False; old best results preserved=None.
- code: selected legacy; promoted=False; old best results preserved=None.
