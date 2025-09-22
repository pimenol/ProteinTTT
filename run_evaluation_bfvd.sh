#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-32-14
#SBATCH --partition=qgpu_free
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate esmfold

# Change dir to $WORK (or other location)
cd "${WORK}/ProteinTTT/ProteinTTT" || exit 1

CHUNK_START=$1
CHUNK_END=$2

echo "Running job for chunk: $CHUNK_START to $CHUNK_END"
# Run script with arguments
python3 run_bfvd.py --chunk_start "$1" --chunk_end "$2"

echo "Job finished successfully."