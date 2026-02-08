#!/bin/bash
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=fgr_proteingym
#SBATCH --output=fgr_proteingym_%j.out
#SBATCH --error=fgr_proteingym_%j.err

# FGR ProteinGym Fitness Prediction Analysis
# 
# Usage:
#   # Run all 10 proteins sequentially:
#   sbatch run_fgr_proteingym.sh
#
#   # Run specific protein by index (for parallel jobs):
#   sbatch run_fgr_proteingym.sh 0
#   sbatch run_fgr_proteingym.sh 1
#   ...
#
#   # Submit all proteins as separate jobs:
#   for i in {0..9}; do sbatch run_fgr_proteingym.sh $i; done

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate proteinttt
else
    source activate proteinttt
fi

# Change to script directory
cd /scratch/project/open-35-8/pimenol1/ProteinTTT_merge/ProteinTTT/scripts

# Configuration file
CONFIG="fgr_proteingym_config.yaml"

# Check if DMS index argument was provided
if [ -n "$1" ]; then
    DMS_INDEX=$1
    echo "Running single protein with DMS index: ${DMS_INDEX}"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Config: ${CONFIG}"
    
    python run_fgr_proteingym.py --config "${CONFIG}" --dms_index "${DMS_INDEX}"
else
    echo "Running all proteins sequentially"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Config: ${CONFIG}"
    
    python run_fgr_proteingym.py --config "${CONFIG}"
fi

echo "Job completed at $(date)"
