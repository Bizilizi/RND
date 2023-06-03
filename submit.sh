#!/bin/bash
#SBATCH --job-name=sbatch_job               # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dzverev@robots.ox.ac.uk # Where to send mail
#SBATCH --time=92:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=48gb                          # Job memory request
#SBATCH --partition=gpu           # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 1 GPUs
#SBATCH --constraint=a6000
# -------------------------------
echo $TIMM_FUSED_ATTN
nvidia-smi
eval "$@"