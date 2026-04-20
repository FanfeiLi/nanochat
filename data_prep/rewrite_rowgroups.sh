#!/bin/bash
export HOME=/lustre/home/fli
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
source /lustre/home/fli/llm_train/nanochat-env/bin/activate

echo "=== Rewriting K5 parquets with row groups ==="
echo "=== Node: $(hostname) ==="
echo "=== Started: $(date) ==="

python /home/fli/llm_train/nanochat/rewrite_rowgroups.py \
    --input /fast/fli/base_data_k5 \
    --output /fast/fli/base_data_k5_rg_full \
    --row-group-size 10000

echo "=== Finished: $(date) ==="
