export PYTHONPATH="$PYTHONPATH:$PWD/src"
#module load cudnn/8.4.1-cu11.6
module load cuda
module load cudnn
#module load cublas
echo "loaded cudnn/8.4.1-cu11.6"
source /lustre/home/fli/llm_train/nanochat-env/bin/activate
echo "activated source from nanochat-env"
echo 'project start'


WANDB_RUN=speedrun bash speedrun_cluster.sh


echo 'project done'