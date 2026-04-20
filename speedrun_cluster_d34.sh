#!/bin/bash

# d34 "~2B-class" training recipe (8xH100 cluster).
# Depth 34, n_embd=2176, 17 heads.
# Parameter counts:
#   - scaling params (matrices + lm_head): 2.00 B  <-- the paper headline
#   - non-embedding (matrices only):       1.93 B
#   - total (incl. wte + value embeds):    3.29 B
# Value-embedding lookups contribute 0 FLOPs, so the compute profile matches
# the 2.00 B scaling-param figure, not the 3.29 B total.
#
# Training horizon (ratio=40, i.e. 2x Chinchilla): 40 * 2.00B ≈ 80.1B tokens
# Dataset at /fast/fli/base_data_k5_aoa14_d4 is ~119 GB → ~66B tokens,
# so this trains for ~1.21 epochs. Multi-epoch is a first-class mode in
# nanochat/dataloader.py (cycles infinitely, tracks epoch counter).
# Matches d26's ratio=40 setting for apples-to-apples cross-depth comparison.
# Expected wall time on 8xH100 @ ~40% MFU: ~90-100 h (≈4 days).

echo "starting speedrun_cluster_d34 script..."
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OMP_NUM_THREADS=1
# IMPORTANT: separate base dir from d26 so concurrent runs don't clobber each
# other's symlinks and checkpoints. /fast/fli/.cache/nanochat is d26's; this
# d34 script uses /fast/fli/.cache/nanochat_d34/ exclusively.
export NANOCHAT_BASE_DIR="/fast/fli/.cache/nanochat_d34"
mkdir -p $NANOCHAT_BASE_DIR

# Symlink tokenizer + eval_bundle to d26's base dir instead of copying.
# Why: cp -rn from d26's eval_bundle hangs because some source files live on
# dead OSTs (4009-4012). Symlinks are metadata-only and don't touch the OSTs
# at link-creation time. d26 reads the same files happily because it has them
# fully resident from previous runs (and apparently doesn't read the bad ones
# during normal eval). Sharing read-only assets via symlink is correct because
# both models use the same vocab and the same eval benchmarks; only the
# training corpus differs (which stays separate via base_data_climbmix).
mkdir -p "$NANOCHAT_BASE_DIR/report" "$NANOCHAT_BASE_DIR/base_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints"
ln -sfn /fast/fli/.cache/nanochat/tokenizer    "$NANOCHAT_BASE_DIR/tokenizer"
ln -sfn /fast/fli/.cache/nanochat/eval_bundle  "$NANOCHAT_BASE_DIR/eval_bundle"

# Symlink the aoa14_d4 rg2k data (row_group_size=2000, ~42 row groups per shard
# for 8-GPU per-file partitioning compatibility)
ln -sfn /fast/fli/base_data_k5_aoa14_d4_rg2k $NANOCHAT_BASE_DIR/base_data_climbmix

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

# Train tokenizer only if we don't already have one cached. Same vocab_size +
# same K5-family data → the cached tokenizer is a valid choice even if it was
# trained on a sibling K5 variant (see discussion in the d26 script).
# NOTE: if you switch to a fundamentally different dataset, delete
# $NANOCHAT_BASE_DIR/tokenizer first so this retrains.
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Reusing cached tokenizer at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
else
    echo "No cached tokenizer found, training a fresh one..."
    python -m scripts.tok_train --max-chars=2000000000
fi

# Always run tok_eval so the report captures the compression ratio on the
# *current* dataset. If bytes/token here drifts far from prior runs, that's a
# signal the tokenizer no longer fits the data distribution.
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
#
# d34 is ~2.00 B scaling params (~2B-class headline, 3.29 B total incl. value embeds).
# ratio=40 = 2x Chinchilla (~80.1 B training tokens ≈ 1.21 epochs over the
# ~66B-token dataset). Matches d26's setting for cross-depth consistency.
# device-batch-size=4 is the memory-safe setting (d34 peaked at ~69.8 GB VRAM).

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=34 \
    --device-batch-size=4 \
    --target-param-data-ratio=40 \
    --save-every=5000 \
    --fp8 \
    --window-pattern=L \
    --run=k5_d34_${RUN_TIMESTAMP}

# evaluate: CORE metric, BPB on train/val, sample generations
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=4

# snapshot pretrain checkpoint (won't be overwritten by reruns)
cp -r $NANOCHAT_BASE_DIR/base_checkpoints $NANOCHAT_BASE_DIR/snapshot_pretrain_d34_${RUN_TIMESTAMP}

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=4 --run=k5_d34_${RUN_TIMESTAMP}
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
cp -r $NANOCHAT_BASE_DIR/chatsft_checkpoints $NANOCHAT_BASE_DIR/snapshot_sft_d34_${RUN_TIMESTAMP}

# -----------------------------------------------------------------------------
# Final report
python -m nanochat.report generate
