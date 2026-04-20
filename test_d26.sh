#!/bin/bash

# Fix PATH for HTCondor minimal environment
export HOME=/lustre/home/fli
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export PYTHONPATH="$PYTHONPATH:$PWD/src"

# torch 2.10 bundles CUDA 12.8 — no module load needed
# NCCL_CUMEM_HOST_ENABLE=0 required for Condor cgroup compatibility
export NCCL_CUMEM_HOST_ENABLE=0

# NCCL watchdog slack: default is 480s (8 min), which is shorter than first-step
# torch.compile tracing + FA3 kernel autotune + dataloader warmup on a cold
# Lustre cache. Extending to 1h keeps step-0 drift from looking like a hang.
# Zero cost in steady state — only changes abort behavior during long stalls.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Debug instrumentation: enable flight recorder so next crash gives us a real
# stack trace of where each rank was stuck, turn on WARN-level NCCL logging,
# and activate the per-rank dbg() statements in base_train.py.
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=WARN
export NANOCHAT_DEBUG=1

nvidia-smi -L 2>/dev/null
source /lustre/home/fli/llm_train/nanochat-env/bin/activate
find /lustre/home/fli/llm_train/nanochat -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

cd /home/fli/llm_train/nanochat
source speedrun_cluster_d26.sh

echo 'project done'
