#!/bin/bash
#SBATCH --job-name=eval-misalign
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/polished-lake/home/jasoncbenn/crosscoder_model_diff_replication/logs/%j.log
#SBATCH --error=/mnt/polished-lake/home/jasoncbenn/crosscoder_model_diff_replication/logs/%j.log
#SBATCH --chdir=/mnt/polished-lake/home/jasoncbenn/crosscoder_model_diff_replication/src/crosscoder_model_diff_replication
#SBATCH --cpus-per-task=4

export HF_HOME=/mnt/polished-lake/home/jasoncbenn/.cache/huggingface/hub/
source /mnt/polished-lake/home/jasoncbenn/crosscoder_model_diff_replication/.venv/bin/activate
python train.py --use_cached_activations --tiny_mode