# Issues found: evaluation noise
| # | Issue | Severity | Status |
|---|---|---|---|
| 01 | [The original classifier predicted "World" for every item (prompt-format letter bias, not padding)](01-classifier-always-world.md) | Critical: invalidated the original run | Fixed in the corrected rerun (new preregistration) |
| 02 | [Environment drift: original setup unavailable, weights re-downloaded, shared CPU](02-environment-drift.md) | Low | Documented |
| 03 | [Overlapping subsets understate the noise of an independent test set](03-design-overlap-understates-noise.md) | Medium (design) | **Addressed by the fresh-sample arm** (20 disjoint fresh sets: 63% [62, 64], slightly above the corrected rerun's 59%, as predicted) |
| 04 | [The decision-statistic CI is narrow because it only covers pool-accuracy uncertainty](04-narrow-ci-meaning.md) | Low (interpretation) | Documented |
| 05 | [The Qwen3-1.7B arm ran in bf16 on CPU; the 0.6B arms ran in fp32](05-qwen3-1.7b-bf16.md) | Low | Documented (preregistered amendment) |
| 06 | [The 1.7B model scored lower than the 0.6B one (82.1 vs 84.5)](06-bigger-model-lower.md) | Low (descriptive) | Documented, not a test |
