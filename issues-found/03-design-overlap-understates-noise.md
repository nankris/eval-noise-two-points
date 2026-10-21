# 03: overlapping subsets of one pool understate the noise of a "new test set"

- The v1 design draws 20 subsets of 200 from the same 1,000-item pool. Subsets share items, so their accuracies are positively correlated, and the hypergeometric (finite-population) variance is smaller than binomial by a factor of (N-n)/(N-1) = 0.80.
- So the pre-registered statistic answers "how different are two subsets of this pool". It doesn't answer "how different are two independent 200-question benchmarks".
- **Fix (strengthening).** v3a uses 20 disjoint 200-item sets from 4,000 fresh test items, with the exact Binomial pair probability. v3b uses a second model (Qwen3-1.7B).
- Status: addressed in v3. See RESULTS.md.
