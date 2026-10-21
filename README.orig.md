# Evaluation noise: is a two-point gain real?

Evaluation noise means a model's measured accuracy can change simply because the evaluation questions change.

Hypothesis: two evaluations of the identical model differ by at least 2 percentage points in at least 25% of pairs.

## Prespecified setup
- Model: Qwen/Qwen3-0.6B, CPU only, evaluation mode, thinking disabled.
- Data: public AG News test split. A fixed random pool of 1,000 articles selected with NumPy seed 126; no label balancing. Evaluate this pool once, then draw 20 independent uniform 200-item subsets without replacement within each subset, using seeds 12600–12619. Subsets may overlap. This is the cheaper subset-noise option authorized by the task; it measures changing questions, not sampling variability on identical questions.
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
- Resumed an existing interrupted evaluation with 148 saved predictions on 2026-10-16; no scoring or analysis settings changed. The original invocation metadata is retained in `results/metadata_before_resume.json`.
- A further resume found no Hugging Face model cache in the current environment. The script now downloads the exact recorded revision into an ignored local model directory and reuses the frozen pool. This changes file loading only, not inference or analysis. An offline startup attempt failed before collecting new predictions.
- During resumed inference, the initial and newly saved predictions all selected World. This diagnostic is reported in the final analysis; the prespecified prompt, model, pool, and scoring were retained. The resulting noise estimate is conditional on this weak classifier.
