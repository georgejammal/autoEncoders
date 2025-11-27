#!/bin/bash
#SBATCH --job-name=testing_env_probe
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --output=/home/ML_courses/03683533_2025/ameer_george_abdallah/checks/env_probe.out
#SBATCH --error=/home/ML_courses/03683533_2025/ameer_george_abdallah/checks/env_probe.err
#SBATCH --time=00:05:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

source /home/ML_courses/03683533_2025/miniconda3/etc/profile.d/conda.sh
conda activate /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38
export PYTHONNOUSERSITE=1
python /home/ML_courses/03683533_2025/ameer_george_abdallah/checks/env_probe.py
python -s /home/ML_courses/03683533_2025/ameer_george_abdallah/checks/env_probe.py