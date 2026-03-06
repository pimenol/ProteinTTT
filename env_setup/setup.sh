#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Require CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Run: module load CUDA/11.7.0"
    exit 1
fi

# Init conda
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Clean slate
if conda env list | grep -q "proteinttt"; then
    echo "Removing existing proteinttt env..."
    conda env remove -n proteinttt -y
fi

# Create env
echo "Creating conda environment..."
conda env create -f "$SCRIPT_DIR/environment.yml"
conda activate proteinttt

# Fix LD_LIBRARY_PATH for systems with old libstdc++ (RHEL/CentOS 8)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
    > "$CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh"
conda deactivate && conda activate proteinttt

# Install OpenFold
echo "Installing OpenFold..."
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# Install lora-diffusion (needs --no-build-isolation for pkg_resources)
echo "Installing lora-diffusion..."
pip install --no-build-isolation 'lora-diffusion @ git+https://github.com/cloneofsimo/lora.git'
pip install numpy==1.26.4  # pin back after lora-diffusion deps

# Install ProteinTTT
echo "Installing ProteinTTT..."
pip install -e "$PROJECT_DIR" --no-deps

echo "Done. Activate with: conda activate proteinttt"
