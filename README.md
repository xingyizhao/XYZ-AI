# XYZ-AI

Local GraphGPO workspace for agentic reinforcement-learning experiments.

## Verified WebShop smoke run

`recipe/GraphGPO` is copied from the upstream `verl-agent` GraphGPO recipe. The
full `verl-agent` runtime and WebShop assets remain outside this repository at
`/scratch/rai/vast1/u6066038/verl-agent-src.ArushA` so this repository contains
only the GraphGPO project files requested for development.

The CHPC launcher runs one complete Qwen2.5-1.5B-Instruct training step:

```bash
cd /scratch/rai/vast1/u6066038/XYZ-AI
./run_webshop_chpc_smoke.sh
```

Verified on an NVIDIA H200 on 2026-08-17. The run completed rollout, GraphGPO
group construction, reward and advantage calculation, reference log-probability
calculation, and the AdamW actor update. The smoke configuration intentionally
uses one training example, two environment rollouts, five WebShop steps, and one
optimizer step; it verifies execution and is not a performance benchmark.

The launcher expects the Python environment at
`/scratch/rai/vast1/u6066038/envs/graphgpo`. Package pins used to create it are
documented in `constraints-webshop.txt`.
