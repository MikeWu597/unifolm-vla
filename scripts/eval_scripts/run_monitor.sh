#!/bin/bash
# ============================================================================
# UnifoLM-VLA-MiniCPMSupervised: Launch monitor server + VLA eval
#
# Start this, then open http://localhost:8778 (via SSH tunnel) to watch.
# ============================================================================
set -e

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/LIBERO}"
export LIBERO_HOME LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-${REPO_ROOT}/models/UnifoLM-VLM-Base}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${REPO_ROOT}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
NUM_TRIALS="${NUM_TRIALS:-3}"
MINICPM_INTERVAL="${MINICPM_INTERVAL:-50}"

echo "============================================"
echo "UnifoLM-VLA-MiniCPMSupervised"
echo "============================================"
echo "LIBERO:       ${LIBERO_HOME}"
echo "VLA ckpt:     ${VLA_CHECKPOINT}"
echo "Task:         ${TASK_SUITE}"
echo "Trials:       ${NUM_TRIALS}"
echo "MiniCPM freq: every ${MINICPM_INTERVAL} steps"
echo "Monitor:      http://localhost:8778"
echo "============================================"

[ ! -f "$VLA_CHECKPOINT" ] && echo "ERROR: checkpoint not found" && exit 1
[ ! -d "$VLM_PRETRAINED_PATH" ] && echo "ERROR: VLM base not found" && exit 1
[ ! -d "$LIBERO_HOME" ] && echo "ERROR: LIBERO not found" && exit 1

export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

conda activate unifolm-vla 2>/dev/null || true

# ── Start monitor server in background ────────────────────────────────
echo "Starting monitor server on port 8778..."
python -m uvicorn unifolm_vla.monitor.server:app --host 0.0.0.0 --port 8778 &
SERVER_PID=$!
sleep 2

# ── Run VLA + MiniCPM ─────────────────────────────────────────────────
echo "Starting VLA evaluation..."
python experiments/LIBERO/eval_libero_monitor.py \
    --args.pretrained-path "$VLA_CHECKPOINT" \
    --args.vlm-pretrained-path "$VLM_PRETRAINED_PATH" \
    --args.task-suite-name "$TASK_SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.unnorm-key "$UNNORM_KEY" \
    --args.minicpm-interval "$MINICPM_INTERVAL" \
    --args.video-out-path "results/monitor_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S)"

# ── Cleanup ──────────────────────────────────────────────────────────
kill $SERVER_PID 2>/dev/null || true
echo "Done."
