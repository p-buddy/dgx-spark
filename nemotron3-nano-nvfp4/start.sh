#!/usr/bin/env bash
set -euo pipefail

# ── Parse parameters encoded in $0 ───────────────────────────────────────────
# Convention: embed flags as KEY-value tokens separated by '__' in the filename.
# The KEY matches the long-form flag name (uppercased, hyphens replacing spaces).
# e.g. serve__PORT-9000.sh
#
# Supported tokens:
#   PORT-<value>    → host port exposed on your machine
#
# Value encoding conventions:
#   Path separators: use '.' in place of '/' (e.g. org.reponame for org/reponame)
#   Hyphens in values are literal (flag boundaries use uppercase KEY- prefix)

_self="$(basename "$0" .sh)"
IFS='__' read -ra _tokens <<< "$_self"

_port=""

for _tok in "${_tokens[@]}"; do
    case "$_tok" in
        PORT-*) _port="${_tok#PORT-}" ;;
    esac
done

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST_PORT="${_port:-8000}"

# ── Fixed settings ────────────────────────────────────────────────────────────
IMAGE="avarok/vllm-dgx-spark:v11"
MODEL="cybermotaz/nemotron3-nano-nvfp4-w4a16"
QUANT="modelopt_fp4"
KV_DTYPE="fp8"
MAX_LEN=131072
GPU_UTIL=0.85
TOOL_CALL_PARSER="qwen3_coder"
REASONING_PARSER="deepseek_r1"

# ── Run ───────────────────────────────────────────────────────────────────────
docker run --rm -it --gpus all --ipc=host \
  -p "${HOST_PORT}:8000" \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${IMAGE}" \
  serve "${MODEL}" \
  --quantization "${QUANT}" \
  --kv-cache-dtype "${KV_DTYPE}" \
  --trust-remote-code \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}"
