#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=./ttt_single_%A.out
#SBATCH --error=./ttt_single_%A.err

OUTPUT_DIR=${1:-""}

source "/scratch/project/open-35-8/pimenol1/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt2

# Use conda's libstdc++ instead of system's (fixes GLIBCXX version issue)
export LD_LIBRARY_PATH="/scratch/project/open-35-8/pimenol1/miniconda3/envs/proteinttt2/lib:$LD_LIBRARY_PATH"

cd /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT || exit 1

# Run the script
if [ -z "$OUTPUT_DIR" ]; then
    python scripts/run_ttt_single.py 
else
    python scripts/run_ttt_single.py  --output_dir "$OUTPUT_DIR"
fi

echo "Job completed at $(date)"

