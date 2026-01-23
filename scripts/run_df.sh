#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/df/df_%A.out
#SBATCH --error=./jobs/df/df_%A.err

# Activate conda environment
source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt2

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1
export PYTHONPATH="${PYTHONPATH}:/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT"
DF_PATH=$1

python3 ./scripts/run_df.py --df_path $DF_PATH

echo "Job finished successfully."