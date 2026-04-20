#!/bin/bash

# Fix PATH for HTCondor minimal environment
export HOME=/lustre/home/fli
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export PYTHONPATH="$PYTHONPATH:$PWD/src"

# torch 2.10 bundles CUDA 12.8 — no module load needed
# NCCL_CUMEM_HOST_ENABLE=0 required for Condor cgroup compatibility
export NCCL_CUMEM_HOST_ENABLE=0

nvidia-smi -L 2>/dev/null
source /lustre/home/fli/llm_train/nanochat-env/bin/activate
find /lustre/home/fli/llm_train/nanochat -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

cd /home/fli/llm_train/nanochat
source speedrun_cluster.sh

echo 'project done'
