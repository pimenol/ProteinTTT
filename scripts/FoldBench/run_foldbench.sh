#!/bin/bash
#SBATCH --job-name=fb_esmfold2
#SBATCH --account=OPEN-37-88
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=./jobs/df/foldbench_%j.out
#SBATCH --error=./jobs/df/foldbench_%j.err

# Usage (from repo root; always set --time so the sum of limits stays inside the FoldBench cap):
#   sbatch --time=HH:MM:SS scripts/FoldBench/run_foldbench.sh --entries foldbench/X.csv --start A --end B --out foldbench/DIR [--screen]

unset PYTHONHOME
unset PYTHONPATH

ROOT=/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
export HF_HOME=/scratch/project/open-35-8/pimenol1/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PY=/scratch/project/open-35-8/pimenol1/miniconda3/envs/esmfold2/bin/python
"$PY" ./scripts/FoldBench/run_foldbench.py --config ./scripts/ESMFold2/M_alpha256_msainput.yaml --root foldbench "$@"
echo "FoldBench run finished."
