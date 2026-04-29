#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=pimenol1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=250GB
#SBATCH --cpus-per-gpu=32
#SBATCH --time=24:00:00
#SBATCH --output=./jobs/benchmark/benchmark_%A.out
#SBATCH --error=./jobs/benchmark/benchmark_%A.err

# Activate conda environment
source "/home/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

cd /home/pimenol1/ProteinTTT|| exit 1
export PYTHONPATH="${PYTHONPATH}:/home/pimenol1/ProteinTTT/"

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
    "${EXTRA_ARGS[@]}"

echo "Benchmark finished."

