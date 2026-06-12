#!/bin/bash
# ============================================================================
# UnifoLM-VLA-ToolCall Agentic LIBERO Evaluation
#
# Dual-head architecture:
#   VLA Head (FlowmatchingActionHead) → actions every step
#   LLM Head (lm_head) → agent reasoning every N steps
#
# Run from anywhere — auto-detects repo root.
# ============================================================================
set -e

# cd to repo root (2 levels up from scripts/eval_scripts/)
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

# ── Auto-fix broken flash-attn ────────────────────────────────────────
# flash_attn .so is often pre-installed but incompatible with torch.
# Uninstall it so QWen2_5.py falls back to sdpa (built-in, always works).
python -c "import flash_attn" 2>/dev/null && {
    echo "Removing pre-installed flash-attn (replaced by sdpa)..."
    pip uninstall flash-attn -y 2>/dev/null || true
} || true

# ── Paths ────────────────────────────────────────────────────────────
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/LIBERO}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-${REPO_ROOT}/models/UnifoLM-VLM-Base}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${REPO_ROOT}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"

# ── Evaluation Config ─────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
NUM_TRIALS="${NUM_TRIALS:-5}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
DEVICE="${DEVICE:-0}"

VLA_INTERVAL="${VLA_INTERVAL:-10}"
MAX_AGENT_ROUNDS="${MAX_AGENT_ROUNDS:-30}"

# Agent VLM: vanilla Qwen2.5-VL for text generation
# Auto-downloads from HF if not cached. Set to local path or 3B variant to save VRAM.
export AGENT_VLM_PATH="${AGENT_VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"

VIDEO_OUT_DIR="results/agent_${TASK_SUITE}/$(date +%Y%m%d_%H%M%S)"

# ── Validate ───────────────────────────────────────────────────────────
echo "============================================"
echo "UnifoLM-VLA-ToolCall Agentic Evaluation"
echo "============================================"
echo "Repo Root:          ${REPO_ROOT}"
echo "LIBERO_HOME:        ${LIBERO_HOME}"
echo "VLA Checkpoint:     ${VLA_CHECKPOINT}"
echo "VLM Base:           ${VLM_PRETRAINED_PATH}"
echo "Task Suite:         ${TASK_SUITE}"
echo "Num Trials/Task:    ${NUM_TRIALS}"
echo "Agent VLM:          ${AGENT_VLM_PATH}"
echo "VLA Interval:       ${VLA_INTERVAL}"
echo "Max Agent Rounds:   ${MAX_AGENT_ROUNDS}"
echo "============================================"

[ ! -f "${VLA_CHECKPOINT}" ] && echo "ERROR: Checkpoint not found: ${VLA_CHECKPOINT}" && exit 1
[ ! -d "${VLM_PRETRAINED_PATH}" ] && echo "ERROR: VLM base not found: ${VLM_PRETRAINED_PATH}" && exit 1
[ ! -d "${LIBERO_HOME}" ] && echo "ERROR: LIBERO not found: ${LIBERO_HOME}" && exit 1

# ── Environment ────────────────────────────────────────────────────────
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# Headless rendering for LIBERO (cloud GPU compatibility)
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# Guard: TensorFlow 2.15 crashes with numpy>=2 → auto-fix
python -c "import numpy; exit(0 if numpy.__version__ < '2' else 1)" 2>/dev/null || {
    echo "Auto-fixing numpy>=2 → numpy<2..."
    pip install "numpy<2" --force-reinstall -q
}

conda activate unifolm-vla 2>/dev/null || true

# ── Run ────────────────────────────────────────────────────────────────
echo ""
echo "Starting agentic evaluation..."
echo ""

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/LIBERO/eval_libero_agent.py \
    --args.pretrained-path "${VLA_CHECKPOINT}" \
    --args.vlm-pretrained-path "${VLM_PRETRAINED_PATH}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${VIDEO_OUT_DIR}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.window-size "${WINDOW_SIZE}" \
    --args.vla-interval "${VLA_INTERVAL}" \
    --args.max-agent-rounds "${MAX_AGENT_ROUNDS}"

echo ""
echo "Done: ${VIDEO_OUT_DIR}"
