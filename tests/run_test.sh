#!/bin/bash
#SBATCH --job-name=test_ttt
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=./jobs/test/test_%A.out
#SBATCH --error=./jobs/test/test_%A.err

# Activate conda environment
source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt2

export PYTHONPATH="${PYTHONPATH}:/scratch/project/open-35-8/pimenol1/ProteinTTT_merge/ProteinTTT"

python tests/test_esmfold_ttt.py

echo "Test finished successfully."
