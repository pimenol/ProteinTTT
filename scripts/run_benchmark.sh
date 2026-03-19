#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/df/benchmark_%A.out
#SBATCH --error=./jobs/df/benchmark_%A.err

# Activate conda environment
source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh || exit 1
export PYTHONPATH="${PYTHONPATH}:/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh"

CONFIG="./scripts/config_benchmark.yaml"
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

python3 ./scripts/run_benchmark.py \
    --config "$CONFIG" \
    --seeds 0 1 2 \
    "${EXTRA_ARGS[@]}"

echo "Benchmark finished."

