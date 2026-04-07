#!/bin/bash
# scripts/azure/setup.sh
# One-time environment setup for the Azure NC48ads A100 v4 VM.
# Run from the home directory as: bash setup.sh
# Idempotent: safe to re-run.

set -euo pipefail

REPO_URL="https://github.com/PatrickAllenCooper/latent_proxy.git"
PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"
HF_CACHE="${HOME}/.cache/huggingface"

log() { echo "[setup] $(date '+%H:%M:%S') $*"; }

# --------------------------------------------------------------------------
# 1. System packages
# --------------------------------------------------------------------------
log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    tmux \
    htop \
    nvtop \
    wget \
    curl \
    build-essential \
    python3-dev \
    libssl-dev \
    ca-certificates

# --------------------------------------------------------------------------
# 2. Verify CUDA
# --------------------------------------------------------------------------
log "Checking CUDA / GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    log "CUDA driver OK"
else
    log "WARNING: nvidia-smi not found. CUDA may not be set up."
fi

# --------------------------------------------------------------------------
# 3. Miniconda
# --------------------------------------------------------------------------
if [ ! -d "${CONDA_DIR}" ]; then
    log "Installing Miniconda..."
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm -f /tmp/miniconda.sh
fi

export PATH="${CONDA_DIR}/bin:${PATH}"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Add to .bashrc if not already there
if ! grep -q "miniconda3" "${HOME}/.bashrc" 2>/dev/null; then
    echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> "${HOME}/.bashrc"
fi

log "Conda: $(conda --version)"

# --------------------------------------------------------------------------
# 4. Clone / update repo
# --------------------------------------------------------------------------
if [ ! -d "${PROJECT_DIR}" ]; then
    log "Cloning repository..."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
else
    log "Updating repository..."
    cd "${PROJECT_DIR}"
    git pull origin main
fi

cd "${PROJECT_DIR}"

# --------------------------------------------------------------------------
# 5. Create / update conda environment
# --------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    log "Conda environment '${ENV_NAME}' already exists; updating..."
    conda activate "${ENV_NAME}"
else
    log "Creating conda environment '${ENV_NAME}' (Python 3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12
    conda activate "${ENV_NAME}"
fi

log "Installing project dependencies..."
pip install -q --upgrade pip
pip install -q -e ".[dev]"

# Ensure PyTorch with CUDA is installed (project lists torch>=2.0 generically)
log "Verifying PyTorch CUDA build..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}  {props.total_memory/1e9:.1f} GB')
"

# --------------------------------------------------------------------------
# 6. Pre-download all Qwen models
# --------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"

log "Pre-downloading Qwen2.5 model weights..."

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    log "  Downloading ${MODEL}..."
    python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL}')
print(f'  Cached: {path}')
"
done

log "All models downloaded."

# --------------------------------------------------------------------------
# 7. Create output directories
# --------------------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/outputs/gamma_scaling"/{1.5B,3B,7B,14B}
mkdir -p "${PROJECT_DIR}/outputs/dpo_study_azure"
mkdir -p "${PROJECT_DIR}/outputs/evaluation_m4_azure"
mkdir -p "${PROJECT_DIR}/outputs/evaluation_m5_azure"
mkdir -p "${PROJECT_DIR}/outputs/generalization_azure"
mkdir -p "${PROJECT_DIR}/outputs/results"
mkdir -p "${PROJECT_DIR}/logs"

# --------------------------------------------------------------------------
# 8. Run test suite
# --------------------------------------------------------------------------
log "Running test suite (CPU tests only)..."
cd "${PROJECT_DIR}"
python -m pytest tests/ -q --tb=line \
    --ignore=tests/test_reward_model.py \
    -k "not test_llm" \
    2>&1 | tail -20

log "=========================================="
log "Setup complete!"
log "Next: cd ${PROJECT_DIR} && tmux new -s experiments"
log "Then: bash scripts/azure/run_all.sh"
log "=========================================="
