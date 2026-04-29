#!/bin/bash
# Qwen3-TTS Benchmark Runner
#
# Compares vllm-omni streaming serving vs HuggingFace transformers offline inference.
# Produces JSON results and comparison plots.
#
# Usage:
#
#   bash run_benchmark.sh
#
#   # Custom settings:
#   NUM_PROMPTS=20 bash run_benchmark.sh
#
# Environment variables:
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 50)
#   MODEL            - Model name (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
#   PORT             - Server port (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GPU_DEVICE="${GPU_DEVICE:-99}"
NUM_PROMPTS="${NUM_PROMPTS:-0}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
PORT="${PORT:-8000}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
RESULT_DIR="${SCRIPT_DIR}/tmp_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-TTS Benchmark"
echo "============================================================"
echo " GPU:          ${GPU_DEVICE}"
echo " Model:        ${MODEL}"
echo " Prompts:      ${NUM_PROMPTS}"
echo " Port:         ${PORT}"
echo " Results:      ${RESULT_DIR}"
echo "============================================================"


# Run HuggingFace baseline benchmark
if [ true ]; then
    echo ""
    echo "============================================================"
    echo " Benchmarking: HuggingFace transformers (offline)"
    echo "============================================================"

    cd "${PROJECT_ROOT}"
    python "${SCRIPT_DIR}/bench_tts_hf.py" \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --num-warmups "${NUM_WARMUPS}" \
        --gpu-device "${GPU_DEVICE}" \
        --config-name "hf_transformers" \
        --result-dir "${RESULT_DIR}"

    sleep 5
fi

# Plot results
echo ""
echo "============================================================"
echo " Generating plots..."
echo "============================================================"

RESULT_FILES=""
LABELS=""


if [ true ]; then
    HF_FILE=$(ls -t "${RESULT_DIR}"/bench_hf_transformers_*.json 2>/dev/null | head -1)
    if [ -n "${HF_FILE}" ]; then
        if [ -n "${RESULT_FILES}" ]; then
            RESULT_FILES="${RESULT_FILES} ${HF_FILE}"
            LABELS="${LABELS} hf_transformers"
        else
            RESULT_FILES="${HF_FILE}"
            LABELS="hf_transformers"
        fi
    fi
fi

if [ -n "${RESULT_FILES}" ]; then
    python "${SCRIPT_DIR}/plot_results.py" \
        --results ${RESULT_FILES} \
        --labels ${LABELS} \
        --output "${RESULT_DIR}/qwen3_tts_benchmark_${TIMESTAMP}.png"
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
echo "============================================================"
