#!/bin/bash
set -e

# Activate proteinttt env using direct path (conda activate may not work in non-interactive shells)
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate proteinttt

PIP="$HOME/miniconda3/envs/proteinttt/bin/pip"

echo "==> Pinning setuptools<70 (restores pkg_resources for torch/openfold)..."
$PIP install "setuptools<70"

echo "==> Installing fair-esm (includes esmfold)..."
$PIP install "fair-esm[esmfold]"

echo "==> Installing dllogger..."
$PIP install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'

echo "==> Cloning and patching openfold (fix CUDA arch + c++14->c++17 for CUDA 12.x)..."
OPENFOLD_TMP="/tmp/openfold_patch_$$"
git clone --quiet https://github.com/aqlaboratory/openfold.git "$OPENFOLD_TMP"
cd "$OPENFOLD_TMP" && git checkout -q 4b41059694619831a7db195b7e0988fc4ff3a307
# Remove unsupported GPU archs (sm_37/52 dropped in CUDA 12) and fix C++ standard
sed -i "s/(3, 7), # K80.*/(7, 0), # V100/" setup.py
sed -i "s/(5, 2), # Titan X//" setup.py
sed -i "s/(6, 1), # GeForce.*/(8, 0), # A100/" setup.py
sed -i "s/-std=c++14/-std=c++17/" setup.py
$PIP install --no-build-isolation "$OPENFOLD_TMP"
# Patch deepspeed.utils.is_initialized() which was removed in deepspeed >0.5.x
OPENFOLD_PRIMITIVES="$HOME/miniconda3/envs/proteinttt/lib/python3.10/site-packages/openfold/model/primitives.py"
sed -i 's/deepspeed\.utils\.is_initialized()/deepspeed.comm.is_initialized() if hasattr(deepspeed.comm, "is_initialized") else False/g' "$OPENFOLD_PRIMITIVES"
rm -rf "$OPENFOLD_TMP"
cd -

echo "==> Installing lora-diffusion (no-build-isolation)..."
$PIP install --no-build-isolation 'lora-diffusion @ git+https://github.com/cloneofsimo/lora.git'

echo "==> Pinning versions overridden by lora-diffusion deps..."
$PIP install \
    "numpy==1.26.4" \
    "diffusers==0.15.0" \
    "transformers==4.30.0" \
    "huggingface-hub==0.16.4" \
    "pytorch-lightning==1.8.4" \
    "torchmetrics==0.10.3" \
    "deepspeed==0.13.4"

echo "==> Installing remaining dependencies..."
$PIP install "biotite==0.40.0" scikit-learn numba pytest

echo "==> Installing ProteinTTT..."
PROJECT_DIR="$(dirname "$(dirname "$0")")"
$PIP install -e "$PROJECT_DIR" --no-deps

echo "Done. Activate with: conda activate proteinttt"
echo "Run tests with: python -m pytest tests/ -v"
