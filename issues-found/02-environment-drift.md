# 02: environment drift between the original machine and the one used for the reruns

- The original setup script and Python environment weren't available for the reruns. I used a different venv (torch 2.14.1+cpu, transformers 5.20.2) instead of the versions in `results/software_versions.json`.
- `model/` had only the tokenizer and config. `model.safetensors` was re-downloaded at the pinned revision `c1899de2…` (a 1.5 GB fp32-equivalent file). It is gitignored and shouldn't be pushed.
- The CPU was shared with other workloads (an ollama server used 4-5 cores at times). Throughput dropped about 10x when two 3-thread jobs ran alongside it, so I paused one job (SIGSTOP/SIGCONT, resumable). Timing numbers are therefore not comparable to the original run's.
- Status: documented. Predictions are deterministic given the weights and prompt, so the drift affects speed, not results. (The transformers version can change logits slightly, which is why the original vs corrected comparison isn't a like-for-like comparison anyway.)
