#!/bin/bash
# ============================================================================
# UnifoLM-VLA LIBERO Simulation Evaluation Script
# ============================================================================
# Usage:
#   1. Run download_weights.py first to download model weights
#   2. Edit the paths below if needed
#   3. Execute: bash run_libero_eval.sh
#
# Override defaults via environment variables:
#   TASK_SUITE=libero_goal NUM_TRIALS=50 bash run_libero_eval.sh
# ============================================================================
set -e

# ===================================================================
# Auto-detect workspace directory (parent of this script)
# ===================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}}"

# ===================================================================
# Path Configuration
# ===================================================================

# LIBERO installation directory
LIBERO_HOME="${LIBERO_HOME:-${WORKSPACE}/LIBERO}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

# UnifoLM-VLM-Base (Qwen2.5-VL base model, HuggingFace format)
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-${WORKSPACE}/models/UnifoLM-VLM-Base}"

# UnifoLM-VLA-LIBERO checkpoint (.pt file)
# Note: config.yaml and dataset_statistics.json must be located
# two directories above the .pt checkpoint:
#   models/UnifoLM-VLA-LIBERO/
#       config.yaml
#       dataset_statistics.json
#       checkpoints/
#           pytorch_model.pt
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${WORKSPACE}/models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt}"

# ===================================================================
# Evaluation Configuration
# ===================================================================

# Task suite (choose one):
#   libero_spatial  - spatial relationship tasks (10 tasks, ~99.0%)
#   libero_object   - object type tasks       (10 tasks, ~100%)
#   libero_goal     - task goal tasks         (10 tasks, ~99.4%)
#   libero_10       - long-horizon tasks      (10 tasks, ~96.2%)
#   libero_90       - full benchmark          (90 tasks)
TASK_SUITE="${TASK_SUITE:-libero_spatial}"

# unnorm_key must match task_suite:
#   libero_spatial -> libero_spatial_no_noops
#   libero_object  -> libero_object_no_noops
#   libero_goal    -> libero_goal_no_noops
#   libero_10      -> libero_10_no_noops
#   libero_90      -> libero_90_no_noops
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"

# Number of episodes per task (quick test=5, full eval=50)
NUM_TRIALS="${NUM_TRIALS:-5}"

# Observation window size (paper default: 2)
WINDOW_SIZE="${WINDOW_SIZE:-2}"

# CUDA device
DEVICE="${DEVICE:-0}"

# Output directory for results and videos
VIDEO_OUT_DIR="results/${TASK_SUITE}/$(date +%Y%m%d_%H%M%S)"

# ===================================================================
# Validate paths
# ===================================================================
echo "============================================"
echo "UnifoLM-VLA LIBERO Evaluation"
echo "============================================"
echo "Workspace:         ${WORKSPACE}"
echo "LIBERO_HOME:       ${LIBERO_HOME}"
echo "VLM Base:          ${VLM_PRETRAINED_PATH}"
echo "VLA Checkpoint:    ${VLA_CHECKPOINT}"
echo "Task Suite:        ${TASK_SUITE}"
echo "Unnorm Key:        ${UNNORM_KEY}"
echo "Num Trials/Task:   ${NUM_TRIALS}"
echo "Window Size:       ${WINDOW_SIZE}"
echo "CUDA Device:       ${DEVICE}"
echo "Output Dir:        ${VIDEO_OUT_DIR}"
echo "============================================"

if [ ! -f "${VLA_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${VLA_CHECKPOINT}"
    echo "Run: python download_weights.py --output-dir ${WORKSPACE}/models"
    exit 1
fi

if [ ! -d "${VLM_PRETRAINED_PATH}" ]; then
    echo "ERROR: VLM base model not found: ${VLM_PRETRAINED_PATH}"
    echo "Run: python download_weights.py --output-dir ${WORKSPACE}/models"
    exit 1
fi

if [ ! -d "${LIBERO_HOME}" ]; then
    echo "ERROR: LIBERO not found: ${LIBERO_HOME}"
    echo "Run setup_vastai.sh first."
    exit 1
fi

# ===================================================================
# torch 2.6+ compatibility: restore old weights_only default
# LIBERO and the VLA checkpoint both use torch.load with numpy objects
# ===================================================================
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# ===================================================================
# Set PYTHONPATH
# ===================================================================
export PYTHONPATH="${LIBERO_HOME}:${WORKSPACE}:${PYTHONPATH}"

# ===================================================================
# Activate conda environment (if available)
# ===================================================================
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate unifolm-vla 2>/dev/null || true
fi

# ===================================================================
# Run evaluation
# ===================================================================
echo ""
echo "Starting evaluation..."
echo ""

CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path "${VLA_CHECKPOINT}" \
    --args.vlm-pretrained-path "${VLM_PRETRAINED_PATH}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${VIDEO_OUT_DIR}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.window-size "${WINDOW_SIZE}"

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: ${VIDEO_OUT_DIR}"
echo "============================================"
