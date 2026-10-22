#!/usr/bin/env python3
"""Prompt-variant diagnosis on DEV items from AG News *train* (disjoint from the test-split pool). 
Usage: diagnose2.py N_DEV"""
import os,sys,json,time
os.environ['OMP_NUM_THREADS']='3'
import torch,numpy as np
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(3)
ROOT=Path(__file__).resolve().parents[1]
N=int(sys.argv[1]); mp=str(ROOT/'model')
tok=AutoTokenizer.from_pretrained(mp);m=AutoModelForCausalLM.from_pretrained(mp,dtype=torch.float32).eval()
tok.padding_side='left'
ds=load_dataset('fancyzhx/ag_news',split='train')
idx=np.random.default_rng(1261).choice(len(ds),N,replace=False)
dev=[dict(text=ds[int(i)]['text'],label=ds[int(i)]['label']) for i in idx]
lab=np.array([d['label'] for d in dev]); print('dev label counts',Counter(lab.tolist()))
def art(d): return tok.decode(tok.encode(d['text'],add_special_tokens=False)[:256])
def ids(words): return [tok.encode(w,add_special_tokens=False)[0] for w in words]
L=ids(list('ABCD')); LS=ids([' A',' B',' C',' D']); NM=ids(['World','Sports','Business','Science']); NMS=ids([' World',' Sports',' Business',' Science'])
chat=lambda c,pre='': tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+pre
LET='Classify this news article. Choose exactly one answer.\nA: World\nB: Sports\nC: Business\nD: Science/Technology\n\nArticle: {a}\n\nAnswer with just A, B, C, or D.'
NAM='Classify this news article into one category: World, Sports, Business, or Science/Technology.\n\nArticle: {a}\n\nAnswer with just the category name.'
variants={
 'original chat letters':(lambda d:chat(LET.format(a=art(d))),L),
 'chat letters + prefill "Answer: "':(lambda d:chat(LET.format(a=art(d)),'Answer:'),LS),
 'chat names':(lambda d:chat(NAM.format(a=art(d))),NM),
 'plain-text letters':(lambda d:'Article: '+art(d)+'\n\nWhich topic is this article about?\nA. World\nB. Sports\nC. Business\nD. Science/Technology\nAnswer:',LS),
 'plain-text names':(lambda d:'Article: '+art(d)+'\n\nTopic (World, Sports, Business, or Science/Technology):',NMS),
 'chat letters, article after options, prefill "The answer is"':(lambda d:chat('Article: '+art(d)+'\n\nWhich topic is this article about?\nA. World\nB. Sports\nC. Business\nD. Science/Technology\n\nAnswer with just the letter.','The answer is'),LS),
}
only=sys.argv[2:] 
for name,(build,cand) in variants.items():
    t=time.time(); res=[]; 
    with torch.inference_mode():
        for o in range(0,N,4):
            inp=tok([build(d) for d in dev[o:o+N and o+4]],return_tensors='pt',padding=True)
            res+=m(**inp,use_cache=False,logits_to_keep=1).logits[:,-1,cand].argmax(-1).tolist()
    res=np.array(res); print(f'{name}: acc={np.mean(res==lab):.3f} preds={dict(sorted(Counter(res.tolist()).items()))} ({time.time()-t:.0f}s)',flush=True)
