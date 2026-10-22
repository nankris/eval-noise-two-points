# Results: is a two-point gain real?

Hypothesis (preregistered): two evaluations of the **same** model on different 200-question test sets differ by at least 2 percentage points in at least 25% of pairs.

**Verdict: PROVEN in all three arms.**

| Arm | Items | Pool acc | Runs: min–max (range) | SD of runs | Binomial SD | Pairs ≥2 pp (observed) | P(diff ≥2 pp) [95% CI] | Verdict |
|---|---:|---:|---|---:|---:|---:|---|---|
| Qwen3-0.6B · 20 overlapping 200-item subsets of a 1,000 pool | 1,000 | 84.5% | 80.5–89.5 (9.0 pp) | 2.35 | 2.56 | 63% of 190 | 59% [56%, 61%] | PROVEN |
| Qwen3-0.6B · 20 disjoint fresh 200-item sets | 4,000 | 84.3% | 76.5–88.5 (12.0 pp) | 2.86 | 2.57 | 63% of 190 | 63% [62%, 64%] | PROVEN |
| Qwen3-1.7B · 20 overlapping subsets of the same pool | 1,000 | 82.1% | 77.5–86.5 (9.0 pp) | 2.16 | 2.71 | 59% of 190 | 61% [59%, 63%] | PROVEN |

Decision statistic: the corrected-pool and Qwen3-1.7B arms use the hypergeometric probability for two 200-item subsets of the 1,000 pool; the fresh-sample arm uses two independent Binomial(200, p) sets. CI = bootstrap over evaluated items (2,000 resamples). Rule: PROVEN if the CI lower bound is ≥ 25%.

**Bigger model check (paired, same 1,000 items):** Qwen3-1.7B 82.1% vs Qwen3-0.6B 84.5%, difference -2.4 pp [95% CI -5.2, +0.3]. They disagree on 19% of items. Detecting a 2 pp difference with 80% power needs about 3,689 items when paired and about 5,460 when unpaired.

**Single-run uncertainty:** the first prespecified 200-item run scored 80.5% with a 95% bootstrap CI of 10.5 pp width (75.0–85.5).

Figures: `figures/01_accuracy_per_set.png`, `02_pairwise_differences.png`, `03_pair_probability.png`, `04_confusion.png`, `00_original_vs_corrected_predictions.png`.

## Limits
- One task (AG News 4-way topic), one prompt, two small Qwen3 models, greedy constrained decoding, CPU.
- The original run was broken (always "World"); the corrected rerun fixed the prompt on train-split dev items before any test predictions (see README).
- The fresh-sample arm finished all 4,000 items (20 disjoint sets) at about 23:46 IST, before the 00:30 IST stopping rule applied.
- The Qwen3-1.7B arm ran in bf16 (amendment 21:40 IST).
- This measures test-set sampling noise only, not rerun nondeterminism.
