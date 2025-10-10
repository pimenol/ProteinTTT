#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=./jobs/fold_%A_%a.out
#SBATCH --error=./jobs/fold_%A_%a.err

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate esmfold

# Change dir to $WORK (or other location)
cd "${WORK}/ProteinTTT/ProteinTTT" || exit 1

CHUNK_START=$1
CHUNK_END=$2
CALCULATE_ONLY_TTT=$3

echo "Running job for chunk: $CHUNK_START to $CHUNK_END with calculate_only_ttt set to $CALCULATE_ONLY_TTT"
# Run script with arguments
python3 run_bfvd.py --chunk_start "$1" --chunk_end "$2" --calculate_only_ttt "$3"

echo "Job finished successfully."