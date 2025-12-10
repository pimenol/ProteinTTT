#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=./jobs/grad_clip_%A.out
#SBATCH --error=./jobs/grad_clip_%A.err

# Activate conda environment

source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt2

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

python3 tune_grad_clip.py --lr "$1" --ags "$2" --grad_clip_max_norm "$3" --lora_rank "$4" --lora_alpha "$5" --step_size "$6"

echo "Job finished successfully."