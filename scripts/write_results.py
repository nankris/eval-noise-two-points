"""Write RESULTS.md from results/*/summary.json and summary_all.json (run after analyze_v2.py)."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]; S={a:json.loads((R/f'results/{a}/summary.json').read_text()) for a in ['v2','v3a','v3b']}
allv=json.loads((R/'results/summary_all.json').read_text()); P=allv.get('paired_v2_vs_v3b')
name={'v2':'Qwen3-0.6B · 20 overlapping 200-item subsets of a 1,000 pool','v3a':'Qwen3-0.6B · %d disjoint fresh 200-item sets'%S['v3a']['n_sets'],'v3b':'Qwen3-1.7B · 20 overlapping subsets of the same pool'}
t='# Results: is a two-point gain real?\n\nHypothesis (pre-registered): two evaluations of the **same** model on different 200-question test sets differ by at least 2 percentage points in at least 25% of pairs.\n\n'
verd=[S[a]['verdict'] for a in S]; t+=f"**Verdict: {'PROVEN in all three arms' if all(v=='PROVEN' for v in verd) else ', '.join(a+': '+S[a]['verdict'] for a in S)}.**\n\n"
t+='| Arm | Items | Pool acc | Runs: min–max (range) | SD of runs | Binomial SD | Pairs ≥2 pp (observed) | P(diff ≥2 pp) [95% CI] | Verdict |\n|---|---:|---:|---|---:|---:|---:|---|---|\n'
for a,s in S.items():
    t+=f"| {name[a]} | {s['n_items']:,} | {s['pool_accuracy_percent']:.1f}% | {s['runs_min']:.1f}–{s['runs_max']:.1f} ({s['runs_max']-s['runs_min']:.1f} pp) | {s['runs_sd_pp']:.2f} | {s['binomial_sd_pp_200']:.2f} | {s['empirical_pair_rate']:.0%} of {s['n_pairs']} | {s['pair_probability_point']:.0%} [{s['pair_probability_95ci'][0]:.0%}, {s['pair_probability_95ci'][1]:.0%}] | {s['verdict']} |\n"
t+=f"\nDecision statistic: v2/v3b use the hypergeometric probability for two 200-item subsets of the 1,000 pool; v3a uses two independent Binomial(200, p) sets. CI = bootstrap over evaluated items (2,000 resamples). Rule: PROVEN if the CI lower bound is ≥ 25%.\n\n"
if P: t+=f"**Bigger model check (paired, same 1,000 items):** Qwen3-1.7B {P['acc_1p7b']:.1f}% vs Qwen3-0.6B {P['acc_0p6b']:.1f}%, difference {P['diff_pp']:+.1f} pp [95% CI {P['diff_95ci_pp'][0]:+.1f}, {P['diff_95ci_pp'][1]:+.1f}]. They disagree on {P['discordant_rate']:.0%} of items. Detecting a 2 pp difference with 80% power needs about {P['n_needed_paired_2pp_80pct_power']:,} items when paired and about {P['n_needed_unpaired_2pp_80pct_power']:,} when unpaired.\n\n"
s=S['v2']; t+=f"**Single-run uncertainty:** the first pre-specified 200-item run scored {s['first_run_accuracy']:.1f}% with a 95% bootstrap CI of {s['first_run_ci_width_pp']:.1f} pp width ({s['first_run_95ci'][0]:.1f}–{s['first_run_95ci'][1]:.1f}).\n\n"
t+='Figures: `figures/01_accuracy_per_set.png`, `02_pairwise_differences.png`, `03_pair_probability.png`, `04_confusion.png`, `00_v1_vs_v2_predictions.png`.\n\n'
t+='## Limits\n- One task (AG News 4-way topic), one prompt, two small Qwen3 models, greedy constrained decoding, CPU.\n- v1 was broken (always "World"); v2 fixed the prompt on train-split dev items before any test predictions (see README).\n- v3a was cut by the pre-registered 00:30 IST stopping rule if incomplete; see `results/v3a/summary.json` for K.\n- v3b ran in bf16 (amendment 21:40 IST).\n- This measures test-set sampling noise only, not rerun nondeterminism.\n'
(R/'RESULTS.md').write_text(t); print(t)
