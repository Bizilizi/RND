#!/bin/bash
#SBATCH --job-name=continual_learning       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dzverev@robots.ox.ac.uk # Where to send mail
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task
#SBATCH --mem=2gb                          # Job memory request
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 1 GPUs
#SBATCH --constraint=p40,m40
# -------------------------------
nvidia-smi
eval "$@"