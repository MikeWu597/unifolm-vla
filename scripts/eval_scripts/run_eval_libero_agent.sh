#!/bin/bash
# ============================================================================
# UnifoLM-VLA-ToolCall Agentic LIBERO Evaluation
# ============================================================================
# Dual-head architecture:
#   VLA Head (FlowmatchingActionHead) → actions every step
#   LLM Head (lm_head) → agent reasoning every N steps
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# ── Paths ────────────────────────────────────────────────────────────
LIBERO_HOME="${LIBERO_HOME:-${WORKSPACE}/LIBERO}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-${WORKSPACE}/models/UnifoLM-VLM-Base}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${WORKSPACE}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"

# ── Evaluation Config ─────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
NUM_TRIALS="${NUM_TRIALS:-5}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
DEVICE="${DEVICE:-0}"

# Agent-specific
VLA_INTERVAL="${VLA_INTERVAL:-10}"    # VLA steps between agent checks
MAX_AGENT_ROUNDS="${MAX_AGENT_ROUNDS:-30}"

VIDEO_OUT_DIR="results/agent_${TASK_SUITE}/$(date +%Y%m%d_%H%M%S)"

# ── Validate ───────────────────────────────────────────────────────────
echo "============================================"
echo "UnifoLM-VLA-ToolCall Agentic Evaluation"
echo "============================================"
echo "Workspace:          ${WORKSPACE}"
echo "VLA Checkpoint:     ${VLA_CHECKPOINT}"
echo "VLM Base:           ${VLM_PRETRAINED_PATH}"
echo "Task Suite:         ${TASK_SUITE}"
echo "Num Trials/Task:    ${NUM_TRIALS}"
echo "VLA Interval:       ${VLA_INTERVAL}"
echo "Max Agent Rounds:   ${MAX_AGENT_ROUNDS}"
echo "Output Dir:         ${VIDEO_OUT_DIR}"
echo "============================================"

if [ ! -f "${VLA_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${VLA_CHECKPOINT}"
    exit 1
fi
if [ ! -d "${VLM_PRETRAINED_PATH}" ]; then
    echo "ERROR: VLM base not found: ${VLM_PRETRAINED_PATH}"
    exit 1
fi
if [ ! -d "${LIBERO_HOME}" ]; then
    echo "ERROR: LIBERO not found: ${LIBERO_HOME}"
    exit 1
fi

# ── Environment ────────────────────────────────────────────────────────
export PYTHONPATH="${LIBERO_HOME}:${WORKSPACE}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate unifolm-vla 2>/dev/null || true
fi

# ── Run ────────────────────────────────────────────────────────────────
echo ""
echo "Starting agentic evaluation..."
echo ""

CUDA_VISIBLE_DEVICES=${DEVICE} python "${WORKSPACE}/experiments/LIBERO/eval_libero_agent.py" \
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
echo "============================================"
echo "Agentic evaluation complete!"
echo "Results saved to: ${VIDEO_OUT_DIR}"
echo "============================================"
