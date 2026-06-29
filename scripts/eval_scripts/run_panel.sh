#!/bin/bash
# ============================================================================
# UnifoLM-VLA-Panel: browser-controlled VLA demo
# ============================================================================
set -e

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# Auto-fix flash-attn / numpy
python -c "import flash_attn" 2>/dev/null && pip uninstall flash-attn -y 2>/dev/null || true
python -c "import numpy; exit(0 if numpy.__version__ < '2' else 1)" 2>/dev/null || pip install "numpy<2" --force-reinstall -q

conda activate unifolm-vla 2>/dev/null || true
pip install -e . --no-deps -q 2>/dev/null || true

echo "============================================"
echo "UnifoLM-VLA-Panel"
echo "============================================"
echo "Open: http://localhost:8778"
echo "============================================"

python examples/panel_demo.py \
    --ckpt "${ROOT}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt" \
    --vlm "${ROOT}/models/UnifoLM-VLM-Base" \
    --port 8778
