#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/fold_%A_%a.out
#SBATCH --error=./jobs/fold_%A_%a.err

# Activate conda environment

source "/scratch/project/open-32-14/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate esmfold

# Change dir to $WORK (or other location)
# cd "${WORK}/ProteinTTT/ProteinTTT" || exit 1
cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

CHUNK_START=$1
CHUNK_END=$2
CALCULATE_ONLY_TTT=$3

echo "Running job for chunk: $CHUNK_START to $CHUNK_END with calculate_only_ttt set to $CALCULATE_ONLY_TTT"
# Run script with arguments
python3 run_bfvd.py --chunk_start "$1" --chunk_end "$2" --calculate_only_ttt "$3" --path_to_df "$4"

echo "Job finished successfully."
