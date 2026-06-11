#!/bin/bash
# ============================================================================
# UnifoLM-VLA vast.ai 一键环境安装脚本
# ============================================================================
# 使用方法:
#   1. 在 vast.ai 租用 GPU 实例 (推荐 RTX 4090 24GB+)
#   2. 选择 CUDA 12.4 的 PyTorch 镜像
#   3. 上传本脚本并执行: bash setup_vastai.sh
# ============================================================================
set -e

echo "============================================"
echo "UnifoLM-VLA Environment Setup for vast.ai"
echo "============================================"

# -------------------------------------------------------------------
# 1. 检查 CUDA 版本
# -------------------------------------------------------------------
echo "[1/8] Checking CUDA version..."
nvidia-smi | head -5
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "WARNING: nvcc not found. Make sure CUDA 12.4 toolkit is available."
fi

# -------------------------------------------------------------------
# 2. 创建 conda 环境
# -------------------------------------------------------------------
echo "[2/8] Creating conda environment..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please use a PyTorch/CUDA template on vast.ai."
    exit 1
fi

conda create -n unifolm-vla python==3.10.18 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unifolm-vla

# -------------------------------------------------------------------
# 3. 克隆仓库
# -------------------------------------------------------------------
echo "[3/8] Cloning UnifoLM-VLA repository..."
if [ ! -d "unifolm-vla" ]; then
    git clone https://github.com/unitreerobotics/unifolm-vla.git
fi
cd unifolm-vla

# -------------------------------------------------------------------
# 4. 安装 PyTorch + 核心依赖 (CUDA 12.4)
# -------------------------------------------------------------------
echo "[4/8] Installing core PyTorch + dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12.4
# torch 2.6+ required: flash-attn's triton rotary uses torch.library.wrap_triton (added in 2.6)
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124

# Core ML dependencies
pip install transformers==4.52.3
pip install accelerate==1.5.2
pip install diffusers==0.35.1
pip install qwen-vl-utils

# General utilities
# numpy<2 is MANDATORY — TensorFlow 2.15 and robosuite need it; some packages pull in 2.x
pip install "numpy>=1.26,<2"
pip install tiktoken einops omegaconf
pip install pydantic==2.10.6
pip install pillow==11.3.0
pip install tyro==0.9.35
pip install scipy matplotlib

# For download_weights.py
pip install huggingface_hub

# Inference server dependencies (optional, for real-robot deployment)
pip install fastapi uvicorn json_numpy

# -------------------------------------------------------------------
# 5. 安装 flash-attn (编译需 5-15 分钟)
# -------------------------------------------------------------------
echo "[5/8] Installing flash-attn (this may take 5-15 minutes)..."
# 确保有 g++ 和 CUDA headers
if ! command -v g++ &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y build-essential
fi
pip install flash-attn==2.5.6 --no-build-isolation || {
    echo "WARNING: flash-attn 2.5.6 failed. Trying newer version..."
    pip install "flash-attn>=2.7.0" --no-build-isolation
}

# -------------------------------------------------------------------
# 6. 注册包 (仅注册路径，不安装训练依赖)
# -------------------------------------------------------------------
echo "[6/8] Registering UnifoLM-VLA package (inference-only mode)..."
# 使用 --no-deps 跳过训练依赖 (deepspeed/wandb/tensorflow/dlimp等推理不需要)
pip install -e . --no-deps

# TensorFlow (CPU-only for image preprocessing tf.image in libero_utils.py)
# 注意: TF 2.15 需要 CUDA 11.8, 但这里只用 CPU 做图像编解码, 不需要 GPU 支持
echo "Installing TensorFlow (CPU-only, for image preprocessing)..."
pip install tensorflow-cpu==2.15.0 2>/dev/null || pip install tensorflow==2.15.0

# -------------------------------------------------------------------
# 7. 安装 LIBERO 仿真环境
# -------------------------------------------------------------------
echo "[7/8] Cloning LIBERO..."
if [ ! -d "../LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
fi

echo "[8/8] Installing LIBERO + all its dependencies..."
pip install -e ../LIBERO

# LIBERO's full runtime dependencies (cumulative from libero_requirements.txt + real usage)
pip install "imageio[ffmpeg]" imageio-ffmpeg
pip install robosuite==1.4.1 robosuite_models
pip install bddl easydict cloudpickle gym mujoco

# torch 2.6+ compatibility: patch LIBERO's torch.load to use weights_only=False
# (torch 2.6 changed default from False → True, breaking old numpy-pickled checkpoints)
echo "Patching LIBERO for torch 2.6+ compatibility..."
grep -rln "torch.load(" ../LIBERO/libero/ --include="*.py" | while read f; do
    sed -i 's/torch\.load(\([^,)]*\))/torch.load(\1, weights_only=False)/g' "$f"
done

# Force numpy<2 — some LIBERO deps may pull in numpy 2.x which breaks TensorFlow
pip install "numpy<2" --force-reinstall 2>/dev/null || true

# Note: mujoco version warning is harmless — LIBERO works with both 3.3.x and 3.9.x

echo ""
echo "============================================"
echo "Setup complete! Verify with:"
echo "  python -c 'from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA; print(\"OK\")'"
echo "============================================"
