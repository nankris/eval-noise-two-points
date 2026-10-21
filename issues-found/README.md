# Issues found: 126 eval noise
| # | Issue | Severity | Status |
|---|---|---|---|
| 01 | [v1 classifier predicted "World" for every item (prompt-format letter bias, not padding)](01-classifier-always-world.md) | Critical: invalidated v1 | Fixed in v2 (re-registered) |
| 02 | [Environment drift: missing setup.sh/mlenv, weights re-downloaded, shared CPU](02-environment-drift.md) | Low | Documented |
| 03 | [Overlapping subsets understate the noise of an independent test set](03-design-overlap-understates-noise.md) | Medium (design) | **Addressed by v3a** (20 disjoint fresh sets: 63% [62, 64], slightly above v2's 59%, as predicted) |
| 04 | [The decision-statistic CI is narrow because it only covers pool-accuracy uncertainty](04-narrow-ci-meaning.md) | Low (interpretation) | Documented |
| 05 | [v3b ran Qwen3-1.7B in bf16 on CPU; v2/v3a ran 0.6B in fp32](05-v3b-bf16.md) | Low | Documented (pre-registered amendment) |
| 06 | [The 1.7B model scored lower than the 0.6B one (82.1 vs 84.5)](06-bigger-model-lower.md) | Low (descriptive) | Documented, not a test |
