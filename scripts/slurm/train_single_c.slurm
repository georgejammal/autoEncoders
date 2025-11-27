#!/bin/bash
#SBATCH --job-name=vae_part_c
#SBATCH --output=/home/ML_courses/03683533_2025/ameer_george_abdallah/logs/%j.out
#SBATCH --error=/home/ML_courses/03683533_2025/ameer_george_abdallah/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=studentkillable

source /home/ML_courses/03683533_2025/miniconda3/etc/profile.d/conda.sh
conda activate /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38
export PYTHONNOUSERSITE=1

# Run the specific ID passed as argument $1
python scripts/train_part_c.py --run-id "$1"