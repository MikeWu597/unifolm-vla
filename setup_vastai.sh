#!/bin/bash
# ============================================================================
# UnifoLM-VLA-ToolCall vast.ai 一键环境安装脚本
# ============================================================================
# Usage:
#   1. Rent GPU on vast.ai (RTX 4090 24GB+, CUDA >=12.4)
#   2. Upload and run: bash setup_vastai.sh
#   3. Download weights: python download_weights.py --output-dir ./models
#   4. Run eval: bash scripts/eval_scripts/run_eval_libero_agent.sh
# ============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo "UnifoLM-VLA-ToolCall Environment Setup"
echo "============================================"

# -------------------------------------------------------------------
# 1. Check CUDA
# -------------------------------------------------------------------
echo "[1/7] Checking CUDA..."
nvidia-smi | head -5
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "WARNING: nvcc not found. Ensure CUDA toolkit is available."
fi

# -------------------------------------------------------------------
# 2. Create conda env
# -------------------------------------------------------------------
echo "[2/7] Creating conda environment..."
if command -v conda &> /dev/null; then
    conda create -n unifolm-vla python==3.10.18 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate unifolm-vla
else
    echo "WARNING: conda not found, using system python"
fi

# -------------------------------------------------------------------
# 3. Core dependencies (exact versions that work together)
# -------------------------------------------------------------------
echo "[3/7] Installing core dependencies..."
pip install --upgrade pip

# Auto-detect CUDA version: 5090/Blackwell needs torch>=2.7 with cu128+
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' 2>/dev/null || echo "12.4")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_SPEC="torch>=2.7"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_SPEC="torch==2.6.0"
fi
echo "CUDA ${CUDA_VER} → installing ${TORCH_SPEC} from ${TORCH_INDEX}"
pip install "${TORCH_SPEC}" torchvision --index-url "${TORCH_INDEX}"

# ML stack
pip install transformers==4.52.3
pip install accelerate==1.5.2
pip install diffusers==0.35.1
pip install qwen-vl-utils

# Utilities
pip install tiktoken einops omegaconf
pip install pydantic==2.10.6 pillow numpy==1.26.4 tyro==0.9.35
pip install scipy matplotlib huggingface_hub

# Inference server (optional)
pip install fastapi uvicorn json_numpy

# -------------------------------------------------------------------
# 4. flash-attn (try pre-built first, source compile as fallback)
# -------------------------------------------------------------------
echo "[4/7] Installing flash-attn..."
pip install flash-attn 2>/dev/null || {
    echo "Pre-built flash-attn failed, compiling from source..."
    pip install flash-attn --no-build-isolation --no-cache-dir
}

# -------------------------------------------------------------------
# 5. Register package (inference-only, skip training deps)
# -------------------------------------------------------------------
echo "[5/7] Registering unifolm_vla package..."
pip install -e . --no-deps

# -------------------------------------------------------------------
# 6. LIBERO + headless rendering
# -------------------------------------------------------------------
echo "[6/7] Installing LIBERO + dependencies..."

# Headless rendering: required for cloud GPU without display
apt-get update -qq && apt-get install -y -qq libosmesa6-dev 2>/dev/null || true
export MUJOCO_GL=osmesa

# TensorFlow CPU for image preprocessing (libero_utils.py uses tf.image)
pip install tensorflow-cpu==2.15.0 2>/dev/null || pip install tensorflow==2.15.0

# LIBERO
if [ ! -d "./LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ./LIBERO
fi
pip install -e ./LIBERO

# LIBERO runtime deps
pip install "imageio[ffmpeg]" imageio-ffmpeg
pip install robosuite==1.4.1 robosuite_models
pip install bddl easydict cloudpickle gym mujoco

# -------------------------------------------------------------------
# 7. Post-install guards
# -------------------------------------------------------------------
echo "[7/7] Finalizing..."

# numpy<2 is MANDATORY — TF 2.15 and robosuite both need it
pip install "numpy<2" --force-reinstall 2>/dev/null || true

# torch 2.6 compat: allow pickle loading of old checkpoints (LIBERO + VLA)
# flash-attn fallback: QWen2_5.py auto-selects sdpa if flash_attn unavailable
# headless rendering: MUJOCO_GL=osmesa for cloud GPU without display

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. python download_weights.py --output-dir ./models"
echo "  2. bash scripts/eval_scripts/run_eval_libero_agent.sh"
echo ""
echo "VLA eval (no agent):"
echo "  bash run_libero_eval.sh"
echo "============================================"
