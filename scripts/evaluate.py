#!/usr/bin/env python3
"""Resumable deterministic AG News pool evaluation; run from experiment root."""
import os, json, time, hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
model_id = 'Qwen/Qwen3-0.6B'
start = time.time()
revision = 'c1899de289a04d12100db370d81485cdf75e47ca'
model_path = snapshot_download(model_id, revision=revision, local_dir=ROOT/'model',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, local_files_only=True).eval()
# Explicit revision fingerprints make cache resolution auditable.
meta = {'model': model_id, 'model_commit': revision,
        'torch': torch.__version__, 'threads': torch.get_num_threads(),
        'dtype': str(next(model.parameters()).dtype), 'device': 'cpu',
        'pool_seed': 126, 'pool_size': 1000, 'batch_size': 4,
        'article_token_cap': 256, 'answer_rule': 'argmax logits restricted to A/B/C/D',
        'thinking': False, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
if (OUT/'pool.json').exists():
    pool = json.loads((OUT/'pool.json').read_text())
    previous = json.loads((OUT/'metadata.json').read_text())
    meta['dataset_fingerprint'] = previous['dataset_fingerprint']
else:
    ds = load_dataset('fancyzhx/ag_news', split='test')
    meta['dataset_fingerprint'] = ds._fingerprint
    indices = np.random.default_rng(126).choice(len(ds), 1000, replace=False).tolist()
    pool = [{'pool_index': i, 'dataset_index': idx, 'text': ds[idx]['text'], 'label': ds[idx]['label']}
            for i,idx in enumerate(indices)]
    (OUT/'pool.json').write_text(json.dumps(pool, indent=2))
meta['pool_sha256'] = hashlib.sha256((OUT/'pool.json').read_bytes()).hexdigest()
(OUT/'metadata.json').write_text(json.dumps(meta, indent=2))
completed = {}
path = OUT/'predictions.jsonl'
if path.exists():
    for line in path.read_text().splitlines():
        row = json.loads(line); completed[row['pool_index']] = row
answer_ids = [tok.encode(x, add_special_tokens=False) for x in 'ABCD']
assert all(len(x)==1 for x in answer_ids), answer_ids
answer_ids = [x[0] for x in answer_ids]
tok.padding_side = 'left'
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
remaining = [p for p in pool if p['pool_index'] not in completed]
with path.open('a', buffering=1) as f, torch.inference_mode():
    for offset in range(0, len(remaining), 4):
        batch = remaining[offset:offset+4]
        prompts = []
        for p in batch:
            article = tok.decode(tok.encode(p['text'], add_special_tokens=False)[:256])
            content = ('Classify this news article. Choose exactly one answer.\n'
                       'A: World\nB: Sports\nC: Business\nD: Science/Technology\n\n'
                       f'Article: {article}\n\nAnswer with just A, B, C, or D.')
            prompts.append(tok.apply_chat_template([{'role':'user','content':content}],
                           tokenize=False, add_generation_prompt=True, enable_thinking=False))
        inputs = tok(prompts, return_tensors='pt', padding=True)
        t = time.time()
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, answer_ids]
        preds = logits.argmax(-1).tolist()
        for j,p in enumerate(batch):
            row = {k:p[k] for k in ['pool_index','dataset_index','label']}
            row.update(prediction=preds[j], correct=int(preds[j]==p['label']),
                       answer_logits=logits[j].tolist(), batch_seconds=time.time()-t)
            f.write(json.dumps(row)+'\n')
        if offset%40==0 or offset+4>=len(remaining):
            print(f'{len(completed)+offset+len(batch)}/1000; elapsed {time.time()-start:.1f}s', flush=True)
meta['execution_seconds_this_invocation'] = time.time()-start
meta['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
(OUT/'metadata.json').write_text(json.dumps(meta, indent=2))
