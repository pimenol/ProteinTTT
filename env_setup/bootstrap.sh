#!/bin/bash
set -e

# Step 1: Ensure conda is available
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        echo "Miniconda already installed. Adding to PATH..."
        export PATH="$HOME/miniconda3/bin:$PATH"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "conda not found. Installing Miniconda..."
        MINICONDA_INSTALLER="$HOME/miniconda.sh"
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$MINICONDA_INSTALLER"
        bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
        rm "$MINICONDA_INSTALLER"
        "$HOME/miniconda3/bin/conda" init bash
        export PATH="$HOME/miniconda3/bin:$PATH"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        echo "Miniconda installed."
    fi
else
    echo "conda already available: $(which conda)"
fi

# Step 2: Load CUDA 12.6
echo "Loading CUDA 12.6.0..."
module load CUDA/12.6.0

# Step 3: Run the main setup
echo "Running setup.sh..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/setup.sh"
