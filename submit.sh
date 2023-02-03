#!/bin/bash
#SBATCH --job-name=continual_learning       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dzverev@robots.ox.ac.uk # Where to send mail
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=12gb                          # Job memory request
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 1 GPUs
#SBATCH --constraint=p40
# -------------------------------
export PATH=/users/dzverev/.conda/envs/cl/bin:$PATH
export PYTHONPATH=\“${BASE}\“:$PYTHONPATH
eval “$(conda shell.bash hook)”
conda activate cl
echo “Starting job w sPAIR”
echo “$(pwd)”
echo “$(nvidia-smi)”
python train.py --model vae-ft \
--config src/vae_ft/configuration/train.ini \
--train_logger wandb \
--evaluation_logger wandb