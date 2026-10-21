#!/usr/bin/env python3
"""v2/v3 resumable evaluation with the fixed prompt (see README v2 re-registration).
usage: evaluate_v2.py --arm v2|v3a|v3b [--threads 3]
  v2 : Qwen3-0.6B (pinned revision, local ./model), 1000-item v1 pool
  v3a: Qwen3-0.6B, 4000 fresh test items (seed 1260, excluding v1 pool)
  v3b: Qwen3-1.7B, 1000-item v1 pool
"""
import os, sys, json, time, argparse, hashlib
ap=argparse.ArgumentParser(); ap.add_argument('--arm',required=True); ap.add_argument('--threads',type=int,default=3); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--dtype',default='float32')
A=ap.parse_args()
os.environ['OMP_NUM_THREADS']=str(A.threads); os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(A.threads)
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/A.arm; OUT.mkdir(parents=True,exist_ok=True)
v1pool=json.loads((ROOT/'results/pool.json').read_text())
if A.arm=='v3a':
    pf=OUT/'pool.json'
    if pf.exists(): pool=json.loads(pf.read_text())
    else:
        ds=load_dataset('fancyzhx/ag_news',split='test'); used={p['dataset_index'] for p in v1pool}
        rest=np.array([i for i in range(len(ds)) if i not in used])
        idx=np.random.default_rng(1260).choice(rest,4000,replace=False).tolist()
        pool=[{'pool_index':i,'dataset_index':int(j),'text':ds[int(j)]['text'],'label':ds[int(j)]['label']} for i,j in enumerate(idx)]
        pf.write_text(json.dumps(pool,indent=1))
else: pool=v1pool
if A.arm=='v3b':
    from huggingface_hub import snapshot_download
    mp=snapshot_download('Qwen/Qwen3-1.7B',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']); mid='Qwen/Qwen3-1.7B'
else: mp=str(ROOT/'model'); mid='Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca'
tok=AutoTokenizer.from_pretrained(mp); tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(mp,dtype=getattr(torch,A.dtype)).eval()
LS=[tok.encode(x,add_special_tokens=False) for x in [' A',' B',' C',' D']]; assert all(len(x)==1 for x in LS),LS; LS=[x[0] for x in LS]
def prompt(t):
    a=tok.decode(tok.encode(t,add_special_tokens=False)[:256])
    c=f'Article: {a}\n\nWhich topic is this article about?\nA. World\nB. Sports\nC. Business\nD. Science/Technology\n\nAnswer with just the letter.'
    return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'The answer is'
meta={'arm':A.arm,'model':mid,'model_path':mp,'n_pool':len(pool),'pool_sha256':hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
      'prompt_example':prompt(pool[0]['text']),'answer_tokens':LS,'threads':A.threads,'dtype':A.dtype,'torch':torch.__version__,'started_ist':time.strftime('%Y-%m-%d %H:%M:%S')}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
path=OUT/'predictions.jsonl'; done=set()
if path.exists():
    for l in path.read_text().splitlines():
        try: done.add(json.loads(l)['pool_index'])
        except Exception: pass
rem=[p for p in pool if p['pool_index'] not in done]
if A.limit: rem=rem[:A.limit]
t0=time.time()
with path.open('a',buffering=1) as f, torch.inference_mode():
    for o in range(0,len(rem),4):
        b=rem[o:o+4]; inp=tok([prompt(p['text']) for p in b],return_tensors='pt',padding=True)
        pos=(inp['attention_mask'].cumsum(-1)-1).clamp(min=0)
        lg=model(**inp,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1,LS]
        pr=lg.argmax(-1).tolist()
        for j,p in enumerate(b):
            f.write(json.dumps({'pool_index':p['pool_index'],'dataset_index':p['dataset_index'],'label':p['label'],'prediction':pr[j],'correct':int(pr[j]==p['label']),'answer_logits':[round(x,4) for x in lg[j].tolist()]})+'\n')
        if (o//4)%25==0: print(f'{A.arm} {len(done)+o+len(b)}/{len(pool)} {time.time()-t0:.0f}s',flush=True)
meta['finished_ist']=time.strftime('%Y-%m-%d %H:%M:%S'); meta['seconds_this_invocation']=time.time()-t0
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2)); print('done',A.arm)
