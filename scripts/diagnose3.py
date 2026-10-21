#!/usr/bin/env python3
"""Diagnose 3 (Grok Bot, 21 Oct): is the all-'World' collapse caused by batching/left-padding, or by the prompt?
Uses the same 60 AG News *train* dev items as diagnose2 (seed 1261). It runs the v1 prompt unbatched (bs=1, no padding)
and batched with explicit position_ids derived from the attention mask."""
import os,sys,time
os.environ['OMP_NUM_THREADS']='3'
import torch,numpy as np
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(3)
ROOT=Path(__file__).resolve().parents[1]; mp=str(ROOT/'model'); N=int(sys.argv[1])
tok=AutoTokenizer.from_pretrained(mp);m=AutoModelForCausalLM.from_pretrained(mp,dtype=torch.float32).eval()
tok.padding_side='left'
ds=load_dataset('fancyzhx/ag_news',split='train')
idx=np.random.default_rng(1261).choice(len(ds),N,replace=False)
dev=[dict(text=ds[int(i)]['text'],label=ds[int(i)]['label']) for i in idx]; lab=np.array([d['label'] for d in dev])
L=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCD']
def prompt(d):
    a=tok.decode(tok.encode(d['text'],add_special_tokens=False)[:256])
    c=('Classify this news article. Choose exactly one answer.\nA: World\nB: Sports\nC: Business\nD: Science/Technology\n\n'f'Article: {a}\n\nAnswer with just A, B, C, or D.')
    return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
print('chat template tail:',repr(prompt(dev[0])[-80:]))
with torch.inference_mode():
    t=time.time(); r1=[]; lg=[]
    for d in dev:
        inp=tok([prompt(d)],return_tensors='pt'); out=m(**inp,use_cache=False,logits_to_keep=1).logits[0,-1]
        r1.append(out[L].argmax().item()); lg.append(out[L].tolist())
        if len(r1)==1: print('top-5 next tokens unbatched item0:',[(tok.decode([i]),round(v,2)) for v,i in zip(*[x.tolist() for x in out.topk(5)])])
    r1=np.array(r1); print(f'unbatched bs=1: acc={np.mean(r1==lab):.3f} preds={dict(Counter(r1.tolist()))} ({time.time()-t:.0f}s)')
    print('mean letter logits', np.mean(lg,0).round(2))
    t=time.time(); r2=[]
    for o in range(0,N,4):
        inp=tok([prompt(d) for d in dev[o:o+4]],return_tensors='pt',padding=True)
        pos=(inp['attention_mask'].cumsum(-1)-1).clamp(min=0)
        r2+=m(**inp,position_ids=pos,use_cache=False,logits_to_keep=1).logits[:,-1,L].argmax(-1).tolist()
    r2=np.array(r2); print(f'batched + position_ids: acc={np.mean(r2==lab):.3f} preds={dict(Counter(r2.tolist()))} ({time.time()-t:.0f}s)',flush=True)
