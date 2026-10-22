# 01: the original classifier predicted "World" for every item

- **Symptom.** All 328 saved predictions from the original run (`results/predictions.jsonl`) are class 0 (World). Accuracy = 95/328 = 29.0%, which is just the World share of those items. The noise estimate from a constant classifier only measures how the class share varies between subsets.
- **Evidence.** `scripts/diagnose2.py` and `scripts/diagnose3.py`, run on 60 AG News *train* items (seed 1261, disjoint from the test pool), show:
  - The original prompt gives 60/60 `A`. Mean letter logits are A 34.2, B 28.0, C 25.0, D 31.5, so A wins by about 3 logits regardless of the article.
  - Unbatched (bs=1, no padding) is identical: 60/60 A. Batched with explicit `position_ids` is also identical. **So it is not a padding or position bug.**
  - The same model with the article first, the options after, and an assistant prefill of "The answer is" scored on ` A..D`: 0.800 accuracy, balanced predictions. Plain-text letters: 0.767.
- **Root cause.** The prompt format. In the original chat turn the options come before the article and the model must emit a bare letter right after `<think>\n\n</think>\n\n`. In that position Qwen3-0.6B's first-option prior dominates. The README's "weak classifier" note in the original run had flagged the symptom but kept the protocol.
- **Fix.** The corrected-prompt preregistration (README, 21:27 IST) changed only the prompt format and answer tokens (` A..D` after the prefill). It was chosen on train-split dev items before any test-pool prediction. Sanity gate passed: pool accuracy 84.5% [82.2, 86.7], max class share 27.1%.
- **Status.** Fixed. The original-run files are untouched; the corrected-rerun outputs are in `results/corrected-pool/`.
