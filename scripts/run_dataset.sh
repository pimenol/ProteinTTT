#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/df/dataset_%A.out
#SBATCH --error=./jobs/df/dataset_%A.err

# Clear inherited Python env vars that break conda activation on this cluster
unset PYTHONHOME
unset PYTHONPATH

# Activate conda environment
source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh || exit 1
export PYTHONPATH="/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh"

# Use the env's python explicitly to avoid PATH ambiguity
PY=/scratch/project/open-35-8/pimenol1/miniconda3/envs/proteinttt/bin/python3

# Diagnostics (visible in .out)
echo "=== Environment diagnostics ==="
echo "which python3: $(which python3)"
echo "PY:            $PY"
echo "PY exists:     $([[ -x "$PY" ]] && echo yes || echo no)"
echo "PY --version:  $("$PY" --version 2>&1)"
echo "CONDA_PREFIX:  $CONDA_PREFIX"
echo "PATH (head):   $(echo "$PATH" | tr ':' '\n' | head -5 | tr '\n' ':')"
echo "==============================="

CONFIG="./scripts/config_dataset.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--config" && -n "$2" ]]; then
        CONFIG="$2"
        shift 2
    else
        EXTRA_ARGS+=("$1")
        shift
    fi
done

"$PY" ./scripts/run_dataset.py \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"

echo "Dataset run finished."
