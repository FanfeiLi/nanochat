#!/bin/bash

# Fix PATH for HTCondor minimal environment
export HOME=/lustre/home/fli
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

export PYTHONPATH="$PYTHONPATH:$PWD/src"
module purge
module load cuda/12.4
module load cudnn/9.10.2

echo "Loaded CUDA 12.4 + cuDNN 9.10.2"
source /lustre/home/fli/llm_train/nanochat-env/bin/activate
echo "activated source from nanochat-env"
echo 'project start'


WANDB_RUN=speedrun bash speedrun_cluster.sh


echo 'project done'