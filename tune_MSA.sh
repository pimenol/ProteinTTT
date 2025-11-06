#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --output=./jobs/msa_lr_%1_ags_%2_grad_clip_max_norm_%3_%A.out
#SBATCH --error=./jobs/msa_lr_%1_ags_%2_grad_clip_max_norm_%3_%A.err

# Activate conda environment

source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

python3 tune_MSA.py --lr "$1" --ags "$2" --grad_clip_max_norm "$3"

echo "Job finished successfully."