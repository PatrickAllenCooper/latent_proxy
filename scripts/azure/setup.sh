#!/bin/bash
# scripts/azure/setup.sh
# One-time environment setup for the Azure NC48ads A100 v4 VM.
# Run from the home directory as: bash ~/latent_proxy/scripts/azure/setup.sh
# Idempotent: safe to re-run after reboots.
#
# IMPORTANT: This script installs NVIDIA drivers and requires a reboot.
# The script detects whether a reboot is needed and exits with instructions.

set -euo pipefail

REPO_URL="https://github.com/PatrickAllenCooper/latent_proxy.git"
PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"
HF_CACHE="${HOME}/.cache/huggingface"

# CUDA 12.4 + driver 550 -- stable, supports A100, available for Ubuntu 24.04
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
DRIVER_VERSION="550"   # nvidia-driver-550 supports A100 on Ubuntu 24.04

log() { echo "[setup] $(date '+%H:%M:%S') $*"; }

# --------------------------------------------------------------------------
# 1. System packages (always needed)
# --------------------------------------------------------------------------
log "Updating apt and installing base packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    tmux \
    htop \
    wget \
    curl \
    build-essential \
    python3-dev \
    libssl-dev \
    ca-certificates \
    pciutils

# --------------------------------------------------------------------------
# 2. NVIDIA drivers + CUDA toolkit (skip if already installed)
# --------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    log "NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    log "NVIDIA driver not found. Installing CUDA ${DRIVER_VERSION} via NVIDIA official repo..."

    # Add CUDA network repo keyring
    wget -q "${CUDA_KEYRING_URL}" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
    sudo apt-get update -qq

    # Install driver and CUDA toolkit
    # nvidia-driver-550 includes the kernel module and CUDA 12.4 runtime
    sudo apt-get install -y -qq \
        "nvidia-driver-${DRIVER_VERSION}" \
        cuda-toolkit-12-4 \
        nvidia-utils-${DRIVER_VERSION}

    log "NVIDIA driver installed. A reboot is required before continuing."
    log ""
    log "  Run:  sudo reboot"
    log "  Then: bash ${PROJECT_DIR}/scripts/azure/setup.sh"
    log ""
    exit 0
fi

# Verify CUDA toolkit is visible to Python (needed for bitsandbytes)
if ! command -v nvcc &>/dev/null; then
    log "Adding CUDA to PATH..."
    CUDA_PATH=$(ls -d /usr/local/cuda-12* 2>/dev/null | head -1 || echo "/usr/local/cuda")
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    # Persist across sessions
    if ! grep -q "cuda" "${HOME}/.bashrc" 2>/dev/null; then
        cat >> "${HOME}/.bashrc" <<BASHEOF
# CUDA
export PATH="${CUDA_PATH}/bin:\${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:\${LD_LIBRARY_PATH:-}"
BASHEOF
    fi
fi

log "CUDA: $(nvcc --version 2>/dev/null | head -1 || echo 'nvcc not found -- runtime only')"

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
    log "Conda environment '${ENV_NAME}' already exists; activating..."
    conda activate "${ENV_NAME}"
else
    log "Creating conda environment '${ENV_NAME}' (Python 3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12
    conda activate "${ENV_NAME}"
fi

log "Installing project dependencies..."
pip install -q --upgrade pip

# Install PyTorch with CUDA 12.4 explicitly before the project install
# (pyproject.toml lists torch>=2.0 generically which may pick a CPU build)
log "Installing PyTorch with CUDA 12.4..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

log "Installing project + dev dependencies..."
pip install -q -e ".[dev]"

# --------------------------------------------------------------------------
# 6. Verify GPU is accessible from Python
# --------------------------------------------------------------------------
log "Verifying PyTorch CUDA access..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}  {props.total_memory/1e9:.1f} GB')
else:
    print('  WARNING: CUDA not available -- check driver/toolkit compatibility')
"

# --------------------------------------------------------------------------
# 7. Pre-download all Qwen models
# --------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"

log "Pre-downloading Qwen2.5 model weights (this may take ~30 min)..."

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
print(f'  Cached at: {path}')
"
done

log "All models downloaded."

# --------------------------------------------------------------------------
# 8. Create output directories
# --------------------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/outputs/gamma_scaling"/{1.5B,3B,7B,14B}
mkdir -p "${PROJECT_DIR}/outputs/dpo_study_azure"
mkdir -p "${PROJECT_DIR}/outputs/evaluation_m4_azure"
mkdir -p "${PROJECT_DIR}/outputs/evaluation_m5_azure"
mkdir -p "${PROJECT_DIR}/outputs/generalization_azure"
mkdir -p "${PROJECT_DIR}/outputs/results"
mkdir -p "${PROJECT_DIR}/logs"

# --------------------------------------------------------------------------
# 9. Run test suite
# --------------------------------------------------------------------------
log "Running test suite (CPU tests only)..."
cd "${PROJECT_DIR}"
python -m pytest tests/ -q --tb=line \
    --ignore=tests/test_reward_model.py \
    -k "not test_llm" \
    2>&1 | tail -20

log "=========================================="
log "Setup complete!"
log ""
log "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
log ""
log "Next steps:"
log "  cd ${PROJECT_DIR}"
log "  tmux new -s experiments"
log "  bash scripts/azure/run_all.sh 2>&1 | tee logs/run_all.log"
log "=========================================="
