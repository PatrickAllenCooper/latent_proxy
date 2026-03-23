#!/bin/bash
# One-time environment setup for CURC Alpine.
# Run from login node: bash scripts/slurm/setup_env.sh

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_BASE="/projects/paco0228/software/anaconda"
ENV_NAME="latent-proxy-env"
HF_CACHE="/scratch/alpine/paco0228/hf_cache"

echo "Setting up latent-proxy environment on CURC..."

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning repository..."
    cd /projects/paco0228
    git clone git@github.com:PatrickAllenCooper/latent_proxy.git
fi

cd "$PROJECT_DIR"
git pull

echo "Creating conda environment: $ENV_NAME"
"$CONDA_BASE/bin/conda" create -n "$ENV_NAME" python=3.12 -y 2>/dev/null || true

source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

echo "Installing dependencies..."
pip install -e ".[dev]"

echo "Creating HuggingFace cache directory..."
mkdir -p "$HF_CACHE"

echo "Creating log directory..."
mkdir -p "$PROJECT_DIR/logs"

echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import gymnasium
import trl
import peft
print('All imports successful.')
"

echo "Setup complete."
