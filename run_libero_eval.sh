#!/bin/bash
# ============================================================================
# UnifoLM-VLA LIBERO Simulation Evaluation
#
# Run from anywhere — auto-detects repo root via script location.
# Override: TASK_SUITE=libero_goal NUM_TRIALS=50 bash run_libero_eval.sh
# ============================================================================
set -e

# cd to repo root (script is at repo root)
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ── Auto-fix broken flash-attn ────────────────────────────────────────
python -c "import flash_attn" 2>/dev/null && {
    echo "Removing pre-installed flash-attn (replaced by sdpa)..."
    pip uninstall flash-attn -y 2>/dev/null || true
} || true

LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/LIBERO}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-${REPO_ROOT}/models/UnifoLM-VLM-Base}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${REPO_ROOT}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
NUM_TRIALS="${NUM_TRIALS:-5}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
DEVICE="${DEVICE:-0}"

VIDEO_OUT_DIR="results/${TASK_SUITE}/$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "UnifoLM-VLA LIBERO Evaluation"
echo "============================================"
echo "Repo Root:          ${REPO_ROOT}"
echo "LIBERO_HOME:        ${LIBERO_HOME}"
echo "VLM Base:           ${VLM_PRETRAINED_PATH}"
echo "VLA Checkpoint:     ${VLA_CHECKPOINT}"
echo "Task Suite:         ${TASK_SUITE}"
echo "Num Trials/Task:    ${NUM_TRIALS}"
echo "============================================"

[ ! -f "${VLA_CHECKPOINT}" ] && echo "ERROR: Checkpoint not found: ${VLA_CHECKPOINT}" && exit 1
[ ! -d "${VLM_PRETRAINED_PATH}" ] && echo "ERROR: VLM base not found: ${VLM_PRETRAINED_PATH}" && exit 1
[ ! -d "${LIBERO_HOME}" ] && echo "ERROR: LIBERO not found: ${LIBERO_HOME}" && exit 1

export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

conda activate unifolm-vla 2>/dev/null || true

echo ""
echo "Starting evaluation..."
echo ""

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/LIBERO/eval_libero.py \
    --args.pretrained-path "${VLA_CHECKPOINT}" \
    --args.vlm-pretrained-path "${VLM_PRETRAINED_PATH}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${VIDEO_OUT_DIR}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.window-size "${WINDOW_SIZE}"

echo ""
echo "Done: ${VIDEO_OUT_DIR}"
