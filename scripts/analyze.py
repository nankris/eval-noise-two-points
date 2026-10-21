#!/usr/bin/env python3
"""Recreate all summary artifacts from saved predictions; no model calls."""
import json, csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'
rows = [json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()]
assert len(rows)==1000 and len({r['pool_index'] for r in rows})==1000, 'Complete the pool first'
rows.sort(key=lambda x:x['pool_index'])
correct = np.array([r['correct'] for r in rows])
run_rows=[]
subset_rows=[]
for run in range(20):
    seed=12600+run
    ids=np.random.default_rng(seed).choice(1000,200,replace=False)
    k=int(correct[ids].sum())
    run_rows.append(dict(run=run+1, seed=seed, n=200, correct=k, accuracy_percent=k/2))
    subset_rows.extend(dict(run=run+1, pool_index=int(i)) for i in ids)
def write_csv(name, data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
write_csv('runs.csv',run_rows)
write_csv('subsets.csv',subset_rows)
pairs=[]
for i in range(20):
    for j in range(i+1,20):
        diff=abs(run_rows[i]['correct']-run_rows[j]['correct'])
        pairs.append(dict(run_a=i+1,run_b=j+1,absolute_difference_pp=diff/2,at_least_2pp=int(diff>=4)))
write_csv('pairs.csv',pairs)
# Bootstrap of binary items is exactly a binomial draw of the resampled count.
rng=np.random.default_rng(1262026)
single_boot=rng.binomial(200,run_rows[0]['correct']/200,size=10000)/2
single_ci=np.percentile(single_boot,[2.5,97.5])
# Each pool bootstrap has K correct items. Evaluate pair probability exactly
# conditional on K using the finite-pool hypergeometric distribution.
ks=rng.binomial(1000,correct.mean(),size=2000)
def pair_probability(k):
    x=np.arange(201)
    pmf=hypergeom.pmf(x,1000,int(k),200)
    # Two tails, using symmetry; ties and differences <=3 are excluded.
    return float(2*np.sum(pmf*hypergeom.cdf(x-4,1000,int(k),200)))
cache={int(k):pair_probability(k) for k in np.unique(ks)}
prob_boot=np.array([cache[int(k)] for k in ks])
prob_ci=np.percentile(prob_boot,[2.5,97.5])
verdict='PROVEN' if prob_ci[0]>=.25 else ('DISPROVEN' if prob_ci[1]<.25 else 'INCONCLUSIVE')
acc=np.array([r['accuracy_percent'] for r in run_rows])
prediction_counts=dict(Counter(str(r['prediction']) for r in rows))
summary=dict(prediction_label_counts=prediction_counts, verdict=verdict, n_pool=1000, pool_correct=int(correct.sum()),
    pool_accuracy_percent=float(correct.mean()*100), n_runs=20,n_per_run=200,
    min_accuracy_percent=float(acc.min()),max_accuracy_percent=float(acc.max()),
    mean_accuracy_percent=float(acc.mean()),sample_std_pp=float(acc.std(ddof=1)),
    headline_range_pp=float(np.ptp(acc)),
    pair_count=190,pairs_at_least_2pp=sum(r['at_least_2pp'] for r in pairs),
    empirical_pair_rate=float(np.mean([r['at_least_2pp'] for r in pairs])),
    conditional_pair_probability=pair_probability(int(correct.sum())),
    pair_probability_bootstrap_95ci=prob_ci.tolist(),
    first_run_accuracy_percent=float(acc[0]),first_run_bootstrap_95ci_percent=single_ci.tolist(),
    first_run_bootstrap_ci_width_pp=float(single_ci[1]-single_ci[0]))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(OUT/'bootstrap_draws.npz',single_accuracy_percent=single_boot,
    pool_correct_counts=ks,pair_probabilities=prob_boot)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].plot(range(1,21),acc,'o-',color='#20639b')
axes[0].set(xlabel='Seeded 200-item subset',ylabel='Accuracy (%)',xticks=[1,5,10,15,20],
    title=f'Identical model, different questions\nObserved range: {np.ptp(acc):.1f} percentage points')
axes[0].axhline(correct.mean()*100,color='gray',linestyle='--',label='1,000-item pool accuracy')
axes[0].legend(fontsize=8)
diffs=[p['absolute_difference_pp'] for p in pairs]
axes[1].hist(diffs,bins=np.arange(-.25,max(diffs)+.76,.5),color='#38a89d',edgecolor='white')
axes[1].axvline(2,color='#c0392b',linestyle='--',label='2-point threshold')
axes[1].set(xlabel='Absolute accuracy difference (percentage points)',ylabel='Pairs of runs',
    title=f"{summary['pairs_at_least_2pp']}/190 pairs differ by ≥2 points ({summary['empirical_pair_rate']:.1%})")
axes[1].legend(fontsize=8)
fig.suptitle('Evaluation noise • Qwen3-0.6B • AG News • deterministic CPU inference',fontsize=12)
fig.tight_layout()
fig.savefig(ROOT/'chart.png',dpi=180)
plt.close(fig)
ci=summary['pair_probability_bootstrap_95ci']
table='\n'.join(f"| {r['run']} | {r['seed']} | {r['correct']}/200 | {r['accuracy_percent']:.1f}% |" for r in run_rows)
writeup=f'''# Evaluation noise: measured result

**Verdict: {verdict}.** The prespecified hypothesis was that at least 25% of pairs of evaluations of an identical model differ by at least two percentage points.

**ONE headline number: {np.ptp(acc):.1f} percentage points.** “The same model 'improved' by {np.ptp(acc):.1f} points just by rerunning it.” Here, “rerunning” means drawing another 200-question evaluation subset, with deterministic predictions. The number is the observed best-minus-worst range over 20 runs; it is not the average gain or a fixed-question stochastic rerun.

| Measure | Result |
|---|---:|
| Model / data | Qwen3-0.6B / AG News test |
| Evaluated pool | 1,000 articles; {int(correct.sum())} correct ({correct.mean():.1%}) |
| Subset evaluations | 20 × 200 articles |
| Prediction counts (0 World, 1 Sports, 2 Business, 3 Sci/Tech) | {prediction_counts} |
| Minimum accuracy | {acc.min():.1f}% |
| Maximum accuracy | {acc.max():.1f}% |
| Mean accuracy | {acc.mean():.2f}% |
| Sample standard deviation | {acc.std(ddof=1):.2f} percentage points |
| Observed pairs with ≥2-point gap | {summary['pairs_at_least_2pp']}/190 ({summary['empirical_pair_rate']:.1%}) |
| Exact pair probability conditional on pool | {summary['conditional_pair_probability']:.1%} |
| 95% pool-item bootstrap CI for pair probability | {ci[0]:.1%}–{ci[1]:.1%} |
| First run accuracy | {acc[0]:.1f}% |
| First run item-bootstrap 95% CI | {single_ci[0]:.1f}%–{single_ci[1]:.1f}% |
| First run CI width | {single_ci[1]-single_ci[0]:.1f} percentage points |
| GPU / Colab CU | None / 0 |

## Plain-English takeaway
Changing which 200 news articles we tested produced a {np.ptp(acc):.1f}-point spread in the score of an unchanged model. A two-point increase on different small test sets can therefore reflect question selection alone; this experiment does not test improvements measured on the same fixed questions.

## Method and interpretation
The README fixed the design and decision rule before inference. Twenty seeds independently select 200 of the same 1,000 articles without replacement within a run. We compute all 190 pairwise gaps, but they are dependent because pairs share runs and subsets overlap. The decision uses a pool-item bootstrap CI for the exact conditional probability of a ≥4-correct-answer gap between two independent hypergeometric subset draws. The lower interval endpoint is {ci[0]:.1%}, compared with the prespecified 25% threshold. This interval captures uncertainty in the pool correctness fraction under iid article sampling; it does not estimate the Monte Carlo error of the observed 190-pair fraction. The single-run interval resamples the first run's 200 correctness labels with replacement 10,000 times; a binomial count draw is mathematically identical to this binary-item bootstrap.

Classifier diagnostic: prediction-label counts are {prediction_counts}. A concentration on one label makes this a weak classifier; the accuracy noise can largely reflect variation in that class's share of each subset. The verdict is conditional on this prespecified scoring pipeline and should not be presented as a measurement of a strong AG News classifier. We retained the protocol rather than selecting a better prompt after inspecting predictions.

The model selects the highest-logit A/B/C/D answer token with thinking disabled and no sampling. This is constrained greedy classification, not unconstrained text generation. Only 1,000 model forward predictions are needed; the 20 subsets reuse those unchanged predictions. Public sources: [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news) and [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B). Model revision, dataset fingerprint, pool hash, software version, and timing are saved in results/metadata.json. The 256-token article cap, prompt, constrained decoding, model capacity, finite pool, and one dataset limit generalization. Overlap makes these subset scores less variable than independent 200-item samples from a very large population. Both sampling reruns and a second dataset were omitted to keep CPU cost low.

## All runs

| Run | Seed | Correct | Accuracy |
|---:|---:|---:|---:|
{table}

## Reproduce

```bash
source /workspace/tools/setup.sh
cd /workspace/work/tig/126-eval-noise-two-points
/workspace/tools/mlenv/bin/python scripts/evaluate.py
/workspace/tools/mlenv/bin/python scripts/analyze.py
```

Evaluation appends predictions after each batch and skips saved pool IDs on resume. Analysis is deterministic and overwrites derived outputs. Raw public pool articles, per-item logits and predictions, subset membership, run scores, pair gaps, bootstrap draws, and summary are in results/.
'''
(ROOT/'writeup.md').write_text(writeup)
print(json.dumps(summary,indent=2))
