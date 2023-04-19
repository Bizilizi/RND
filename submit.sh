#!/bin/bash
#SBATCH --job-name=continual_learning       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dzverev@robots.ox.ac.uk # Where to send mail
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=12gb                          # Job memory request
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 1 GPUs
#SBATCH --constraint=p40
# -------------------------------
nvidia-smi
eval "$@"

sbatch submit.sh python train.py --config src/transformer_vq_vae/configuration/train.ini --model transformer-vq-vae --wandb_dir /scratch/shared/beegfs/dzverev/wand --max_epochs 1000 --encoder_mlm_loss_loss_weight 0 --decoder_regression_loss_loss_weight 0 --embeddings_distance l2
