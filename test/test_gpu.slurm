#!/bin/bash
#SBATCH --job-name=george_test_gpu
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --output=/home/ML_courses/03683533_2025/ameer_george_abdallah/test/test_gpu.out
#SBATCH --error=/home/ML_courses/03683533_2025/ameer_george_abdallah/test/test_gpu.err
#SBATCH --time=00:05:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

source /home/ML_courses/03683533_2025/miniconda3/etc/profile.d/conda.sh
conda activate /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38
python test_gpu.py
