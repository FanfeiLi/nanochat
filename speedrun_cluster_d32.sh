#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat

echo "starting speedrun script..."
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/fast/fli/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
# Symlink K5 data as the pretraining dataset
ln -sfn /fast/fli/base_data_k5 $NANOCHAT_BASE_DIR/base_data_climbmix

# -----------------------------------------------------------------------------
# Python venv setup with uv
#
# install uv (if not already installed)
#command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
#[ -d ".venv" ] || uv venv
# install the repo dependencies
#uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
#module load cudnn/8.4.1-cu11.6


# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
#curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# K5 data already symlinked at $NANOCHAT_BASE_DIR/base_data_climbmix (288 shards)
# No download needed — train tokenizer directly on K5 data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d32 model is ~1.86B parameters (calibrated from d34=2.22B).
# With target-param-data-ratio=12, needs ~480 shards. K5 dataset has plenty.
# d34 peaked at 69.8GB with device-batch-size=4, so d32 uses batch-size=4 too.

# pretrain the d32 model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=32 --device-batch-size=4 --save-every=5000 --fp8 --run=k5_d32
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=4
# snapshot pretrain checkpoint (won't be overwritten by reruns)
cp -r $NANOCHAT_BASE_DIR/base_checkpoints $NANOCHAT_BASE_DIR/snapshot_pretrain_${RUN_TIMESTAMP}

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --minimal --device-batch-size=4 --run=k5_d32
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
# snapshot SFT checkpoint
cp -r $NANOCHAT_BASE_DIR/chatsft_checkpoints $NANOCHAT_BASE_DIR/snapshot_sft_${RUN_TIMESTAMP}

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
