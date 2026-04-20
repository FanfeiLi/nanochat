#!/bin/bash

# d26 "1B-class" training recipe (8xH100 cluster).
# Depth 26, n_embd=1664, 13 heads.
# Parameter counts:
#   - scaling params (matrices + lm_head): 0.918 B  <-- the "~1B model" headline
#   - non-embedding (matrices only):       0.864 B
#   - total (incl. wte + value embeds):    1.682 B
# Value-embedding lookups contribute 0 FLOPs, so the compute profile matches
# the 0.92 B scaling-param figure, not the 1.68 B total.
#
# Training horizon (ratio=40, i.e. 2x Chinchilla): 40 * 0.918B ≈ 36.7B tokens
# Dataset has ~19.5B tokens available, so this trains for ~1.88 epochs.
# Multi-epoch is a first-class mode in nanochat/dataloader.py (cycles infinitely
# and tracks an epoch counter). At <4 repetitions, repeated tokens are nearly as
# valuable as unique tokens (Muennighoff et al. 2023, arXiv:2305.16264).
# Matches d34's ratio=40 setting so cross-depth comparisons stay consistent.
# Expected wall time on 8xH100 @ ~40% MFU: ~32-40 h

echo "starting speedrun_cluster_d26 script..."
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/fast/fli/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Symlink the K5 v2 zipf/p95/d4 data (rg2k variant: row_group_size=2000 so each
# shard has ~15 row groups, enough for 8-GPU per-file row-group partitioning)
ln -sfn /fast/fli/base_data_k5_v2_aoa_zipf_p95_k5_d4_rg2k $NANOCHAT_BASE_DIR/base_data_climbmix

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Reset the report directory with a fresh header section
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
# Build the rustbpe Tokenizer (cheap no-op rebuild if already built)
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Train tokenizer only if we don't already have one cached. Same data + same
# vocab_size → same tokenizer, so we can safely reuse across depth runs.
# NOTE: if you switch to a *different* dataset, delete $NANOCHAT_BASE_DIR/tokenizer
# first so this retrains on the new data.
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Reusing cached tokenizer at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
else
    echo "No cached tokenizer found, training a fresh one..."
    python -m scripts.tok_train --max-chars=2000000000
fi

# Always run tok_eval so the report captures the compression ratio on the
# *current* dataset. Useful sanity check when reusing a tokenizer trained on
# a different data mix — if bytes/token here drifts far from the old K5
# number, that's a signal the tokenizer no longer fits the distribution.
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
#
# d26 is ~0.92B scaling params (~1B-class headline, 1.68B total incl. value embeds).
# ratio=40 = 2x Chinchilla (~36.7B training tokens ≈ 1.88 epochs over the
# 19.5B-token dataset). Matches d34's setting and removes the tight 1x-Chinchilla
# data-margin failure mode.
# device-batch-size=4 mirrors d32; d26 is smaller so this has headroom.

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --device-batch-size=4 \
    --target-param-data-ratio=40 \
    --save-every=5000 \
    --fp8 \
    --window-pattern=L \
    --run=k5_d26_${RUN_TIMESTAMP}

# evaluate: CORE metric, BPB on train/val, sample generations
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=4

# snapshot pretrain checkpoint (won't be overwritten by reruns)
cp -r $NANOCHAT_BASE_DIR/base_checkpoints $NANOCHAT_BASE_DIR/snapshot_pretrain_d26_${RUN_TIMESTAMP}

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=4 --run=k5_d26_${RUN_TIMESTAMP}
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
cp -r $NANOCHAT_BASE_DIR/chatsft_checkpoints $NANOCHAT_BASE_DIR/snapshot_sft_d26_${RUN_TIMESTAMP}

# -----------------------------------------------------------------------------
# Final report
python -m nanochat.report generate
