#!/usr/bin/env python3
"""Analysis for three arms: corrected-pool (corrected prompt, 0.6B, 1000 pool), fresh-disjoint-sets (0.6B, 4000 fresh items, 20 disjoint sets), qwen3-1.7b (1.7B, 1000 pool).
No model calls. Writes results/<arm>/summary.json, runs.csv, pairs.csv, results/summary_all.json, figures/*.png"""
import json, csv, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import hypergeom, binom, norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
NAMES=['World','Sports','Business','Sci/Tech']
def load(arm):
    p=R/arm/'predictions.jsonl'
    if not p.exists(): return None
    rows={}
    for l in p.read_text().splitlines(): r=json.loads(l); rows[r['pool_index']]=r
    return [rows[k] for k in sorted(rows)]
def wcsv(path,data):
    with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def p_indep(p,n=200,thr=4):
    x=np.arange(n+1); pmf=binom.pmf(x,n,p); return float(2*np.sum(pmf*binom.cdf(x-thr,n,p)))
def p_hyper(N,K,n=200,thr=4):
    x=np.arange(n+1); pmf=hypergeom.pmf(x,N,K,n); return float(2*np.sum(pmf*hypergeom.cdf(x-thr,N,K,n)))
def n_unpaired(p,delta=.02,alpha=.05,power=.8):
    z=norm.ppf(1-alpha/2)+norm.ppf(power); return int(np.ceil(z**2*2*p*(1-p)/delta**2))
def n_paired(discord,delta=.02,alpha=.05,power=.8):
    z=norm.ppf(1-alpha/2)+norm.ppf(power); return int(np.ceil(z**2*discord/delta**2))
def verdict(ci): return 'PROVEN' if ci[0]>=.25 else ('DISPROVEN' if ci[1]<.25 else 'INCONCLUSIVE')
out={}
for arm in ['corrected-pool','fresh-disjoint-sets','qwen3-1.7b']:
    rows=load(arm)
    exp_n=(200 if arm=='fresh-disjoint-sets' else 1000)
    if rows is None or len(rows)<exp_n: out[arm]={'status':f'incomplete {0 if rows is None else len(rows)}/{exp_n}'}; continue
    c=np.array([r['correct'] for r in rows]); N=len(c); pred=np.array([r['prediction'] for r in rows]); lab=np.array([r['label'] for r in rows])
    cm=np.zeros((4,4),int)
    for a,b in zip(lab,pred): cm[a,b]+=1
    share=np.bincount(pred,minlength=4)/N
    gate=bool(c.mean()>=.5 and share.max()<=.6)
    runs=[]
    K=20 if arm!='fresh-disjoint-sets' else min(20,N//200)
    for k in range(K):
        if arm=='fresh-disjoint-sets': ids=np.arange(k*200,(k+1)*200)  # disjoint sets of a random (seed 1260) sample
        else: ids=np.random.default_rng(12600+k).choice(1000,200,replace=False)
        runs.append(dict(run=k+1,correct=int(c[ids].sum()),accuracy_percent=float(c[ids].sum()/2)))
    acc=np.array([r['accuracy_percent'] for r in runs]); cnt=np.array([r['correct'] for r in runs])
    pairs=[dict(run_a=i+1,run_b=j+1,abs_diff_pp=abs(cnt[i]-cnt[j])/2,at_least_2pp=int(abs(cnt[i]-cnt[j])>=4)) for i in range(K) for j in range(i+1,K)]
    (R/arm).mkdir(exist_ok=True); wcsv(R/arm/'runs.csv',runs); wcsv(R/arm/'pairs.csv',pairs)
    rng=np.random.default_rng(1262026)
    ks=rng.binomial(N,c.mean(),size=2000)  # item bootstrap of binary items == binomial draw of the count
    if arm=='fresh-disjoint-sets':
        cache={int(k):p_indep(k/N) for k in np.unique(ks)}; point=p_indep(c.mean()); stat='P(two independent 200-item sets differ by >=4 correct), Binomial(200,p)'
    else:
        cache={int(k):p_hyper(N,int(k)) for k in np.unique(ks)}; point=p_hyper(N,int(c.sum())); stat='P(two 200-item subsets of the 1000 pool differ by >=4 correct), hypergeometric'
    boot=np.array([cache[int(k)] for k in ks]); ci=np.percentile(boot,[2.5,97.5]).tolist()
    single=rng.binomial(200,cnt[0]/200,size=10000)/2; sci=np.percentile(single,[2.5,97.5]).tolist()
    p=c.mean()
    s=dict(arm=arm,n_items=N,pool_accuracy_percent=float(p*100),pool_accuracy_95ci_percent=(np.percentile(rng.binomial(N,p,10000)/N*100,[2.5,97.5])).tolist(),
        sanity_gate_passed=gate,prediction_share={NAMES[i]:float(share[i]) for i in range(4)},confusion_matrix_rows_true_cols_pred=cm.tolist(),
        per_class_recall={NAMES[i]:float(cm[i,i]/cm[i].sum()) for i in range(4)},
        runs_min=float(acc.min()),runs_max=float(acc.max()),runs_range_pp=float(np.ptp(acc)),runs_mean=float(acc.mean()),runs_sd_pp=float(acc.std(ddof=1)),
        binomial_sd_pp_200=float(np.sqrt(p*(1-p)/200)*100),
        n_sets=K,n_pairs=len(pairs),pairs_at_least_2pp=int(sum(x['at_least_2pp'] for x in pairs)),empirical_pair_rate=float(np.mean([x['at_least_2pp'] for x in pairs])),
        decision_statistic=stat,pair_probability_point=point,pair_probability_95ci=ci,verdict=verdict(ci) if gate else 'NOT ANALYZED (sanity gate failed)',
        textbook_independent_pair_probability=p_indep(p),
        first_run_accuracy=float(acc[0]),first_run_95ci=sci,first_run_ci_width_pp=float(sci[1]-sci[0]),
        n_needed_unpaired_2pp_80pct_power=n_unpaired(p))
    out[arm]=s; (R/arm/'summary.json').write_text(json.dumps(s,indent=2))
# paired comparison 0.6B (corrected-pool) vs 1.7B (qwen3-1.7b) on identical items
a,b=load('corrected-pool'),load('qwen3-1.7b')
if a and b and len(a)==len(b)==1000:
    ca=np.array([r['correct'] for r in a]); cb=np.array([r['correct'] for r in b]); disc=float(np.mean(ca!=cb))
    rng=np.random.default_rng(7); bs=[(cb[i]-ca[i]).mean()*100 for i in [rng.integers(0,1000,1000) for _ in range(10000)]]
    out['paired_corrected_vs_qwen3_1p7b']=dict(acc_0p6b=float(ca.mean()*100),acc_1p7b=float(cb.mean()*100),diff_pp=float((cb-ca).mean()*100),diff_95ci_pp=np.percentile(bs,[2.5,97.5]).tolist(),
        discordant_rate=disc,n_needed_paired_2pp_80pct_power=n_paired(disc),n_needed_unpaired_2pp_80pct_power=n_unpaired((ca.mean()+cb.mean())/2))
(R/'summary_all.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
# figures
plt.rcParams.update({'font.size':11})
arms=[x for x in ['corrected-pool','fresh-disjoint-sets','qwen3-1.7b'] if 'verdict' in out.get(x,{})]
lbl={'corrected-pool':'Qwen3-0.6B, overlapping subsets of 1,000','fresh-disjoint-sets':'Qwen3-0.6B, %s disjoint fresh sets (%s items)'%(out.get('fresh-disjoint-sets',{}).get('n_sets','?'),format(out.get('fresh-disjoint-sets',{}).get('n_items',0),',')),'qwen3-1.7b':'Qwen3-1.7B, overlapping subsets of 1,000'}
if arms:
    fig,ax=plt.subplots(1,len(arms),figsize=(5.2*len(arms),4.2),squeeze=False)
    for i,arm in enumerate(arms):
        rr=list(csv.DictReader(open(R/arm/'runs.csv'))); y=[float(r['accuracy_percent']) for r in rr]
        ax[0,i].plot(range(1,len(y)+1),y,'o-',color='#20639b'); ax[0,i].axhline(out[arm]['pool_accuracy_percent'],ls='--',color='gray')
        ax[0,i].set(title=f"{lbl[arm]}\nrange {out[arm]['runs_range_pp']:.1f} pp, SD {out[arm]['runs_sd_pp']:.2f} pp",xlabel='200-question test set #',ylabel='Accuracy (%)')
        ax[0,i].title.set_fontsize(10)
    fig.suptitle('Same model, same prompt, different 200 questions'); fig.tight_layout(); fig.savefig(F/'01_accuracy_per_set.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(1,len(arms),figsize=(5.2*len(arms),4.2),squeeze=False)
    for i,arm in enumerate(arms):
        d=[float(r['abs_diff_pp']) for r in csv.DictReader(open(R/arm/'pairs.csv'))]
        ax[0,i].hist(d,bins=np.arange(-.25,max(d)+.76,.5),color='#38a89d',edgecolor='white'); ax[0,i].axvline(1.75,color='#c0392b',ls='--',label='≥2 pp')
        ax[0,i].set(title=f"{lbl[arm]}\n{out[arm]['pairs_at_least_2pp']}/{out[arm]['n_pairs']} pairs differ by ≥2 pp",xlabel='|accuracy difference| (pp)',ylabel='pairs'); ax[0,i].title.set_fontsize(10); ax[0,i].legend()
    fig.tight_layout(); fig.savefig(F/'02_pairwise_differences.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2)); xs=np.arange(len(arms))
    pts=[out[a]['pair_probability_point'] for a in arms]; lo=[out[a]['pair_probability_95ci'][0] for a in arms]; hi=[out[a]['pair_probability_95ci'][1] for a in arms]
    ax.errorbar(xs,pts,yerr=[np.array(pts)-lo,np.array(hi)-pts],fmt='o',capsize=6,color='#20639b',label='decision statistic, 95% CI')
    ax.scatter(xs,[out[a]['empirical_pair_rate'] for a in arms],marker='x',color='#e67e22',label='observed share of pairs',zorder=3)
    ax.axhline(.25,color='#c0392b',ls='--',label='preregistered 25% bar'); ax.set_xticks(xs,[lbl[a].replace(', ','\n') for a in arms],fontsize=8); ax.set_ylim(0,1)
    ax.set_ylabel('P(two evals differ by ≥2 pp)'); ax.legend(fontsize=8); ax.set_title('How often does an unchanged model "gain" 2 points?'); fig.tight_layout(); fig.savefig(F/'03_pair_probability.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(1,len(arms),figsize=(4.6*len(arms),4),squeeze=False)
    for i,arm in enumerate(arms):
        cm=np.array(out[arm]['confusion_matrix_rows_true_cols_pred']); ax[0,i].imshow(cm,cmap='Blues')
        for r in range(4):
            for q in range(4): ax[0,i].text(q,r,cm[r,q],ha='center',va='center',color='white' if cm[r,q]>cm.max()/2 else 'black')
        ax[0,i].set(xticks=range(4),yticks=range(4),xticklabels=NAMES,yticklabels=NAMES,xlabel='predicted',ylabel='true',title=f"{lbl[arm].replace(', ','\n')}\nacc {out[arm]['pool_accuracy_percent']:.1f}%")
    fig.tight_layout(); fig.savefig(F/'04_confusion.png',dpi=170); plt.close(fig)
# original (broken) vs corrected prediction distribution
orig=[json.loads(l) for l in (R/'predictions.jsonl').read_text().splitlines()]
if 'corrected-pool' in arms:
    corr=load('corrected-pool'); fig,ax=plt.subplots(figsize=(7,3.8)); w=.38; x=np.arange(4)
    c1=np.bincount([r['prediction'] for r in orig],minlength=4)/len(orig); c2=np.bincount([r['prediction'] for r in corr],minlength=4)/len(corr); tl=np.bincount([r['label'] for r in corr],minlength=4)/len(corr)
    ax.bar(x-w/2,c1,w,label=f'original prompt (n={len(orig)}), acc {np.mean([r["correct"] for r in orig])*100:.1f}%',color='#c0392b'); ax.bar(x+w/2,c2,w,label=f'corrected prompt (n=1000), acc {out["corrected-pool"]["pool_accuracy_percent"]:.1f}%',color='#20639b')
    ax.scatter(x,tl,marker='_',s=900,color='black',label='true label share',zorder=3); ax.set_xticks(x,NAMES); ax.set_ylabel('share of predictions'); ax.legend(fontsize=8); ax.set_title('The bug: the original prompt answered "World" every time')
    fig.tight_layout(); fig.savefig(F/'00_original_vs_corrected_predictions.png',dpi=170); plt.close(fig)
