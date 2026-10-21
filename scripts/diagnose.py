#!/usr/bin/env python3
"""Root-cause diagnosis on the first 40 pool items: compare variants of the v1 pipeline."""
import os, json, sys, time
os.environ['OMP_NUM_THREADS']='3'
import torch, numpy as np
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(3)
ROOT=Path(__file__).resolve().parents[1]
mp=str(ROOT/'model')
tok=AutoTokenizer.from_pretrained(mp, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(mp, dtype=torch.float32, local_files_only=True).eval()
pool=json.loads((ROOT/'results/pool.json').read_text())[:40]
ids=[tok.encode(x,add_special_tokens=False) for x in 'ABCD']; ids=[i[0] for i in ids]
print('ids',ids,[tok.decode([i]) for i in ids])
def content(p):
    a=tok.decode(tok.encode(p['text'],add_special_tokens=False)[:256])
    return ('Classify this news article. Choose exactly one answer.\nA: World\nB: Sports\nC: Business\nD: Science/Technology\n\n'
            f'Article: {a}\n\nAnswer with just A, B, C, or D.')
def run(name, build, bs=1, pad='left'):
    tok.padding_side=pad
    res=[]; t=time.time()
    with torch.inference_mode():
        for o in range(0,len(pool),bs):
            b=pool[o:o+bs]
            inp=tok([build(p) for p in b],return_tensors='pt',padding=True)
            lg=model(**inp).logits[:,-1,ids]
            res+=lg.argmax(-1).tolist()
    lab=[p['label'] for p in pool]
    print(f'{name}: acc={np.mean(np.array(res)==np.array(lab)):.3f} preds={dict(Counter(res))} ({time.time()-t:.0f}s)',flush=True)
chat=lambda p,th: tok.apply_chat_template([{'role':'user','content':content(p)}],tokenize=False,add_generation_prompt=True,enable_thinking=th)
print(repr(chat(pool[0],False)[-120:]))
#run('v1 prompt, bs=1 (no batching)', lambda p:chat(p,False), bs=1)
#run('v1 prompt, bs=4 left pad', lambda p:chat(p,False), bs=4)

# --- top-token inspection and variants
with torch.inference_mode():
    for p in pool[:6]:
        inp=tok(chat(p,False),return_tensors='pt'); lg=model(**inp).logits[0,-1]
        top=lg.topk(5); print('label',p['label'],[(tok.decode([i]),round(v,1)) for v,i in zip(top.values.tolist(),top.indices.tolist())])
# variant: label-word scoring (first token of each class name), same chat prompt style but names instead of letters
names=['World','Sports','Business','Science']
nids=[tok.encode(n,add_special_tokens=False)[0] for n in names]; print('name ids',nids,[tok.decode([i]) for i in nids])
def content_names(p):
    a=tok.decode(tok.encode(p['text'],add_special_tokens=False)[:256])
    return ('Classify this news article into one category: World, Sports, Business, or Science/Technology.\n\n'
            f'Article: {a}\n\nAnswer with just the category name.')
def run2(name, build, cand, bs=4):
    tok.padding_side='left'; res=[]; t=time.time()
    with torch.inference_mode():
        for o in range(0,len(pool),bs):
            b=pool[o:o+bs]; inp=tok([build(p) for p in b],return_tensors='pt',padding=True)
            res+=model(**inp).logits[:,-1,cand].argmax(-1).tolist()
    lab=[p['label'] for p in pool]
    print(f'{name}: acc={np.mean(np.array(res)==np.array(lab)):.3f} preds={dict(Counter(res))} ({time.time()-t:.0f}s)',flush=True)
run2('names, chat nothink', lambda p: tok.apply_chat_template([{'role':'user','content':content_names(p)}],tokenize=False,add_generation_prompt=True,enable_thinking=False), nids)
# reversed letter assignment to test positional bias: A=Sci/Tech, B=Business, C=Sports, D=World
def content_rev(p):
    a=tok.decode(tok.encode(p['text'],add_special_tokens=False)[:256])
    return ('Classify this news article. Choose exactly one answer.\nA: Science/Technology\nB: Business\nC: Sports\nD: World\n\n'
            f'Article: {a}\n\nAnswer with just A, B, C, or D.')
tok.padding_side='left'; res=[]
with torch.inference_mode():
    for o in range(0,40,4):
        b=pool[o:o+4]; inp=tok([tok.apply_chat_template([{'role':'user','content':content_rev(p)}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for p in b],return_tensors='pt',padding=True)
        res+=model(**inp).logits[:,-1,ids].argmax(-1).tolist()
mapped=[3-r for r in res]; lab=[p['label'] for p in pool]
print('reversed letters: pred letters',dict(Counter(res)),'acc',np.mean(np.array(mapped)==np.array(lab)))
