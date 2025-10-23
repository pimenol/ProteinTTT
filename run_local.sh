#!/bin/bash

# Check if two arguments (start and end) were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_local.sh <chunk_start> <chunk_end>"
    exit 1
fi

echo "--- Starting job with chunk $1 to $2 ---"

# 1. Activate your conda environment
# Make sure this path is correct for your local machine
source "/scratch/project/open-32-14/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate esmfold

# 2. Change to the correct working directory
# Make sure this path exists on your local machine
cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

# 3. Run the Python script with the provided arguments
python3 calculate_metrics_script.py --chunk_start "$1" --chunk_end "$2"

echo "--- Job finished successfully for chunk $1 to $2 ---"