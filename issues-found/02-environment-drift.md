# 02: environment drift between the original sandbox and this box

- `/workspace/tools/setup.sh` and `/workspace/tools/mlenv` don't exist here. I used the shared venv `/workspace/tig-env` (torch 2.14.1+cpu, transformers 5.20.2) instead of the versions in `results/software_versions.json`.
- `model/` had only the tokenizer and config. `model.safetensors` was re-downloaded at the pinned revision `c1899de2…` (a 1.5 GB fp32-equivalent file). It is gitignored and shouldn't be pushed.
- The box CPU is shared with other workloads (an ollama server used 4-5 cores at times). Throughput dropped about 10x when two 3-thread jobs ran alongside it, so I paused one job (SIGSTOP/SIGCONT, resumable). Timing numbers are therefore not comparable to v1's.
- Status: documented. Predictions are deterministic given the weights and prompt, so the drift affects speed, not results. (The transformers version can change logits slightly, which is why v1 vs v2 isn't a like-for-like comparison anyway.)
