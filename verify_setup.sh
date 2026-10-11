#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  [${GREEN}PASS${NC}] $1"; }
fail() { echo -e "  [${RED}FAIL${NC}] $1"; }

echo "=== 1. Host NVIDIA & Driver ==="
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    pass "Host GPU detected: $GPU_NAME"
else
    fail "Host nvidia-smi is failing or NVIDIA driver is not loaded."
fi

echo "=== 2. Docker & Permissions ==="
if docker ps &> /dev/null; then
    pass "Docker daemon is active and running without sudo."
else
    fail "Cannot connect to Docker daemon (check systemctl status docker or docker group)."
fi

echo "=== 3. CDI Specification ==="
if [ -f /etc/cdi/nvidia.yaml ] && nvidia-ctk cdi list 2>&1 | grep -q "nvidia.com/gpu"; then
    pass "CDI spec (/etc/cdi/nvidia.yaml) is present and recognized."
else
    fail "CDI spec missing or invalid. Run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
fi

echo "=== 4. Docker GPU Passthrough ==="
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    pass "Docker container GPU passthrough (--gpus all) works."
elif docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    pass "Docker container GPU passthrough works via CDI (--device nvidia.com/gpu=all)."
else
    fail "Docker container cannot access the GPU."
fi

echo "=== 5. Virtual Environment & DVC ==="
if [[ "$VIRTUAL_ENV" != "" ]]; then
    pass "Virtual environment active: $VIRTUAL_ENV"
else
    fail "Virtual environment is not active. Run: source .venv/bin/activate"
fi

if command -v dvc &> /dev/null && dvc version &> /dev/null; then
    pass "DVC is installed and functioning without missing modules."
else
    fail "DVC is not working or not installed in current environment."
fi

if [ -d .dvc ] && [ -f data.dvc ] && [ -f models.dvc ]; then
    pass "DVC initialized and data/ / models/ pointer files exist."
else
    fail "DVC repository structure missing. Check .dvc directory and pointer files."
fi

echo "=== 6. Git Tracking ==="
if git rev-parse --is-inside-work-tree &> /dev/null; then
    if git check-ignore -q .venv; then
        pass ".venv is correctly ignored by Git."
    else
        fail ".venv is not ignored by Git (add .venv/ to .gitignore)."
    fi
    pass "Git repository is initialized."
else
    fail "Not inside a Git repository."
fi

echo "==============================="
