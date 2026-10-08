#!/bin/bash
#SBATCH --job-name=esmc_jacobian
#SBATCH --account=OPEN-37-88
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=07:00:00
#SBATCH --output=./jobs/df/esmc_jacobian_%j.out
#SBATCH --error=./jobs/df/esmc_jacobian_%j.err

# Usage (from the checkout that holds this script):
#   sbatch scripts/ESMFold2/run_contact_jacobian.sh --config X.yaml --csv S.csv --msa_dir DIR --out DIR [--start N --end M]
unset PYTHONHOME PYTHONPATH
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="$PWD"
export HF_HOME=/scratch/project/open-35-8/pimenol1/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/project/open-35-8/pimenol1/miniconda3/envs/esmfold2/bin/python scripts/ESMFold2/contact_jacobian.py "$@"
echo "Jacobian run finished."
