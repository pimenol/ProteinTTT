#!/bin/bash
#SBATCH --job-name=e2_meanw
#SBATCH --account=OPEN-37-88
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/jobs/df/esmfold2_meanw_%j.out
#SBATCH --error=/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/jobs/df/esmfold2_meanw_%j.err

# Usage: CODE=<repo or worktree root> sbatch scripts/ESMFold2/run_weight_avg.sh --config X.yaml --out DIR [--seed S --start N --end M]
unset PYTHONHOME PYTHONPATH
CODE=${CODE:-/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh}
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
export HF_HOME=/scratch/project/open-35-8/pimenol1/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/project/open-35-8/pimenol1/miniconda3/envs/esmfold2/bin/python scripts/ESMFold2/weight_avg.py "$@"
echo "Weight-averaging run finished."
