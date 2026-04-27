#!/bin/bash
# Example Slurm job script for running the ProteinTTT pytest suite on a GPU node.
# Please adapt this file to your Slurm configuration (account/partition/resources),
# and update the environment activation + paths for your cluster.
#SBATCH --job-name=test_ttt
#SBATCH --account=pimenol1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=./jobs/test/test_%A.out
#SBATCH --error=./jobs/test/test_%A.err
#SBATCH --nodelist=h200-06
#SBATCH --exclusive

# Activate conda environment
source "/home/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

export PYTHONPATH="${PYTHONPATH}:/home/pimenol1/ProteinTTT/"

python -m pytest -ra tests

echo "Test finished."
