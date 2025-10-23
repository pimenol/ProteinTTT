#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --output=./jobs_eval/eval_%A_%a.out
#SBATCH --error=./jobs_eval/eval_%A_%a.err

# Activate conda environment

source "/scratch/project/open-32-14/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate esmfold

# Change dir to $WORK (or other location)
# cd "${WORK}/ProteinTTT/ProteinTTT" || exit 1
cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

python3 calculate_metrics_script.py --chunk_start "$1" --chunk_end "$2"

echo "Job finished successfully."
