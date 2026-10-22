# Evaluation noise: is a two-point gain real?

**Result (final, 21 Oct 2026 23:46 IST): PROVEN in all three arms.** Two 200-question evaluations of the *same* model differ by ≥2 pp with probability 59–63% (lower CI bounds 56–62%, against a 25% bar). See [RESULTS.md](RESULTS.md). The original run (an always-"World" classifier) was invalid and was replaced by a new preregistration for the corrected runs.

Evaluation noise means a model's measured accuracy can change simply because the evaluation questions change.

Hypothesis: two evaluations of the identical model differ by at least 2 percentage points in at least 25% of pairs.

## Prespecified setup
- Model: Qwen/Qwen3-0.6B, CPU only, evaluation mode, thinking disabled.
- Data: public AG News test split. A fixed random pool of 1,000 articles selected with NumPy seed 126; no label balancing. Evaluate this pool once, then draw 20 independent uniform 200-item subsets without replacement within each subset, using seeds 12600–12619. Subsets may overlap. This is the cheaper subset-noise option; it measures changing questions, not sampling variability on identical questions.
- Prompt: classify into World, Sports, Business, or Science/Technology. Choose the highest next-token logit among A/B/C/D (deterministic constrained greedy decoding, one answer token). Exact label match determines correctness. Article tokenization is capped at 256 tokens before adding the instruction.
- Report 20 accuracies (min, max, sample standard deviation), all 190 unordered pairwise absolute differences, and the fraction at least 2 percentage points. Integer correct-count differences of at least 4 implement the threshold exactly.
- Headline number: maximum minus minimum accuracy across these 20 runs, explicitly labeled as the observed range, not a typical gain.
- Single-run uncertainty: 10,000 item bootstrap resamples for the FIRST prespecified run, percentile 95% CI and width.
- Pair-rate uncertainty: bootstrap the 1,000 evaluated pool items 2,000 times and recompute the conditional probability that two independent uniform 200-item subsets of that pool differ by at least four correct answers. Use the hypergeometric distribution to compute each probability exactly. This avoids treating the 190 overlapping pairs as independent observations. This interval reflects uncertainty in the pool's correctness rate under an iid article assumption, not new inference runs.

## Fixed decision rule
PROVEN if the lower endpoint of that 95% bootstrap interval for the pair probability exceeds or equals 25%; DISPROVEN if its upper endpoint is below 25%; otherwise INCONCLUSIVE. The empirical fraction among the 190 pairs is reported separately, and will not replace the decision statistic.

## Execution and limits
Scripts save raw predictions incrementally and skip completed items on resume. CPU only; GPU: none; Colab CU used: 0. No paid APIs. Results are conditional on this model, prompt, public test pool, and sample size. Pool overlap reduces variability relative to independent samples from a very large population. No claim about fixed-question stochastic reruns will be made.

## Execution notes
- Runtime dataset identifier: `fancyzhx/ag_news` (the canonical name). The short `ag_news` alias failed with this installed datasets/hub version before any predictions were collected.
- An initial model metadata lookup used an attribute absent from the installed Transformers version; it was replaced with the cached snapshot revision. Model files were loaded from the local Hugging Face cache to avoid remote metadata delays.
- These startup fixes did not change the prespecified pool, prompts, seeds, scoring, or analysis. Full package versions are in `results/software_versions.json` and `requirements.txt`.
- Resumed an interrupted evaluation with 148 saved predictions on 2026-10-16; no scoring or analysis settings changed. The original invocation metadata is retained in `results/metadata_before_resume.json`.
- A further resume found no Hugging Face model cache on the machine. The script now downloads the exact recorded revision into an ignored local model directory and reuses the frozen pool. This changes file loading only, not inference or analysis. An offline startup attempt failed before collecting new predictions.
- During resumed inference, the initial and newly saved predictions all selected World. This diagnostic is reported in the final analysis; the prespecified prompt, model, pool, and scoring were retained. The resulting noise estimate is conditional on this weak classifier.

## Corrected-prompt preregistration: fixing the broken classifier (written 21 Oct 2026 21:27 IST, before any corrected-run test-pool predictions)
**Why.** The original run predicted `World` (A) for all 328 saved items, so its accuracy equals the World share of the pool. A diagnosis on 60 AG News **train** items (seed 1261, disjoint from the test pool) found the cause (`scripts/diagnose2.py`, `scripts/diagnose3.py`):
- It is **not** a batching or padding bug. Unbatched bs=1 and batched with explicit position_ids both give 60/60 `A`, the same as the original run.
- It is the **prompt format**. The chat-template prompt lists the options before the article and asks "Answer with just A, B, C, or D". On it Qwen3-0.6B has a huge prior for `A` (mean letter logits A 34.2, B 28.0, C 25.0, D 31.5), so it never picks anything else. Other formats on the same 60 dev items:

| Prompt variant (dev, n=60) | Acc | Prediction spread |
|---|---|---|
| original chat prompt, options before article, score `A..D` | 0.167 | all A |
| chat + prefill "Answer:" | 0.267 | mostly D |
| chat, category names | 0.267 | all Sports |
| plain text (no chat template), article then options, "Answer:" + ` A..D` | 0.767 | balanced |
| **chat, article then options, prefill "The answer is", score ` A..D`** | **0.800** | balanced |

**Corrected-run change (one thing only).** I use the last prompt format, picked on train-split dev items only and before any test-pool prediction. The user turn is `Article: <article, 256-token cap>\n\nWhich topic is this article about?\nA. World\nB. Sports\nC. Business\nD. Science/Technology\n\nAnswer with just the letter.`. The chat template keeps thinking disabled. The assistant turn is prefilled with `The answer is` and scored by argmax over the next-token logits of ` A`, ` B`, ` C`, ` D`. **Unchanged:** the model and revision, the 1000-item pool (`results/pool.json`), the subset seeds 12600-12619, the 200-item subsets, the 2 pp (>=4 correct) threshold, the hypergeometric bootstrap, and the decision rule. The original-run files stay untouched. The corrected-rerun outputs are in `results/corrected-pool/`.
**Sanity gate (pre-set):** the corrected rerun is analyzed only if pool accuracy is >= 50% and no class gets > 60% of predictions. Otherwise it is reported as still broken.

## Strengthening arms (preregistered 21 Oct 2026 21:27 IST, before running)
The original and corrected-rerun designs draw overlapping subsets from one 1000-item pool. That understates the noise you get from a truly new test set of the same size. Two extra arms:
- **Fresh-sample arm: independent test sets.** 4000 fresh AG News test items, not in the original pool (numpy seed 1260 over the remaining test indices), split into **20 disjoint 200-item test sets**. All 190 pairs are then independent. Same corrected prompt and model. Statistic: the fraction of the 190 pairs whose correct counts differ by >= 4. Decision statistic: P(two independent 200-item sets differ by >= 4), computed exactly from Binomial(200, p) x Binomial(200, p). Its 95% CI comes from bootstrapping the 4000 items (2000 resamples). PROVEN / DISPROVEN / INCONCLUSIVE use the same 25% rule.
- **Qwen3-1.7B arm: a second model.** Qwen/Qwen3-1.7B on the same 1000 pool with the same corrected prompt, analyzed with the corrected-rerun pipeline. This checks the claim on a model that isn't the 0.6B. Same sanity gate.
- **Textbook check (descriptive).** For each arm I compare against the binomial prediction: SE of a 200-item accuracy = sqrt(p(1-p)/200), and P(|X1-X2| >= 4) for two independent Binomial(200, p) runs. I also report the n needed for a 2 pp difference to be significant (two-sided alpha .05, 80% power) for unpaired and paired designs. The paired n uses the observed discordance between the 0.6B (corrected rerun) and 1.7B models on the same items.
Prediction (written before running): with p around 0.8 the binomial SD of a 200-item accuracy is about 2.8 pp, so P(diff >= 2 pp) should be about 0.5-0.6 for independent sets. I expect PROVEN for the fresh-sample arm and for the corrected rerun, with the corrected rerun slightly lower because the subsets overlap.
- *Amendment, 21:40 IST, before any Qwen3-1.7B result was inspected.* The Qwen3-1.7B arm loads Qwen3-1.7B in **bfloat16** on CPU, not fp32. fp32 would need about 7 GB of RAM, and the machine had about 5 GB free. Revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. This affects numerics only. The prompt, pool and scoring are unchanged.
- *Amendment, 21:50 IST, blind (no fresh-sample or Qwen3-1.7B prediction has been inspected or scored yet).* The machine is heavily loaded by unrelated workloads (load average about 25 on 8 cores), and both arms are running at about 10 items/min. Stopping rule, fixed now: if the fresh-sample arm hasn't finished by **00:30 IST 22 Oct**, it is analyzed on the first K = floor(n_done/200) complete disjoint sets, in pool order, with all K(K-1)/2 pairs, and the same decision statistic, which depends only on the pool accuracy over the n_done items. the Qwen3-1.7B arm is analyzed only if it finishes all 1,000. Otherwise it is reported as incomplete.
