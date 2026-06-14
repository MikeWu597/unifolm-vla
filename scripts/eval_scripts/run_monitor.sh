#!/bin/bash
# ============================================================================
# UnifoLM-VLA-MiniCPMSupervised — Dual conda env launcher
#
# minicpm  env: MiniCPM-o-4.5 + web server (transformers 4.51.0)
# unifolm-vla env: VLA + LIBERO eval               (transformers 4.52.3)
# ============================================================================
set -e

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

# ── Paths ────────────────────────────────────────────────────────────
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/LIBERO}"
export LIBERO_HOME LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
VLM_PATH="${VLM_PRETRAINED_PATH:-${REPO_ROOT}/models/UnifoLM-VLM-Base}"
VLA_CKPT="${VLA_CHECKPOINT:-${REPO_ROOT}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
NUM_TRIALS="${NUM_TRIALS:-3}"
MINICPM_INTERVAL="${MINICPM_INTERVAL:-50}"
PORT=8778

echo "============================================"
echo "UnifoLM-VLA-MiniCPMSupervised"
echo "============================================"
echo "Task:         ${TASK_SUITE}  Trials: ${NUM_TRIALS}"
echo "MiniCPM freq: every ${MINICPM_INTERVAL} steps"
echo "Monitor:      http://localhost:${PORT}"
echo "============================================"

[ ! -f "$VLA_CKPT" ] && echo "ERROR: checkpoint not found" && exit 1

# ── Setup MiniCPM conda env (once) ───────────────────────────────────
MINICPM_ENV="${CONDA_PREFIX:-/venv/minicpm}"
if ! conda info --envs 2>/dev/null | grep -q minicpm; then
    echo "Creating minicpm conda environment..."
    conda create -n minicpm python==3.10.18 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate minicpm
pip install -q "setuptools<70" accelerate "transformers==4.51.0" 2>/dev/null || true
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
pip install -q fastapi uvicorn pillow numpy 2>/dev/null || true
pip install -q "minicpmo-utils[all]>=1.0.5" librosa soundfile 2>/dev/null || true
pip install -e . --no-deps -q 2>/dev/null || true
pip install "numpy<2" --force-reinstall -q 2>/dev/null || true
conda activate unifolm-vla
pip install -e . --no-deps -q 2>/dev/null || true
pip install -q requests 2>/dev/null || true
pip install "numpy<2" --force-reinstall -q 2>/dev/null || true

# CRLF cleanup (Windows Git → Linux)
find "${REPO_ROOT}" -name "*.py" -exec sed -i 's/\r$//' {} \; 2>/dev/null || true

# ── Kill any old server ──────────────────────────────────────────────
fuser -k ${PORT}/tcp 2>/dev/null || true

# ── Start MiniCPM server (in minicpm env) ───────────────────────────
echo "Starting MiniCPM server (conda: minicpm)..."
conda activate minicpm
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export HF_TOKEN="${HF_TOKEN:-}"
python -m uvicorn unifolm_vla.monitor.server:app --host 0.0.0.0 --port ${PORT} &
SERVER_PID=$!
sleep 3
echo "Server PID: ${SERVER_PID}"

# ── Run VLA (in unifolm-vla env) ────────────────────────────────────
echo "Starting VLA evaluation (conda: unifolm-vla)..."
conda activate unifolm-vla
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

python experiments/LIBERO/eval_libero_monitor.py \
    --args.pretrained-path "$VLA_CKPT" \
    --args.vlm-pretrained-path "$VLM_PATH" \
    --args.task-suite-name "$TASK_SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.unnorm-key "$UNNORM_KEY" \
    --args.minicpm-interval "$MINICPM_INTERVAL" \
    --args.video-out-path "results/monitor_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S)"

# ── Cleanup ──────────────────────────────────────────────────────────
kill $SERVER_PID 2>/dev/null || true
echo "Done."
