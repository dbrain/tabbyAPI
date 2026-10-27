#!/bin/bash
# Flash-Next (Swift1.5 EXL3 4.05) on exllamav3 1.6.5 + Kyojin kernels (dbrain/exllamav3 strix-point) behind
# TabbyAPI (dbrain/tabbyAPI strix-point) on Strix Point (gfx1150, 96 GB). Started by exl3-flashnext.service via
# the drop-ins next to this file (install as /etc/systemd/system/<unit>.d/kyojin.conf); same port, API key and
# model id as the llama.cpp flashnext.service it replaces. usage: serve.sh [port]
set -euo pipefail
P=/home/dbrain/dev/tinyfiddler/exl3-prod
PORT="${1:-8765}"
CTX="${EXL3_CTX:-262144}"
CFG="$P/tabbyAPI/config.yml"
# EXL3_TREE points the server at another exllamav3 build (A/B of a dev tree through this unit)
TREE="${EXL3_TREE:-$P/exllamav3}"
DEPLOY="$(cd "$(dirname "$0")" && pwd)"

export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm EXL3_ROCM_SDK=/opt/rocm
# Without streaming the n-gram table is loaded into RAM and the box swaps
export EXL3_NGRAM_STREAM=1 EXL3_NGRAM_POPULATE=0
# Token embedding rows read from the file on lookup: -0.85 GB anonymous RSS, decode/prefill unchanged (2026-10-27)
export EXL3_EMBED_STREAM="${EXL3_EMBED_STREAM-1}"
# Two-stage draft head, packed for Swift's head with tools/qwen/draft2s_pack.py from 42K captured draft steps
# (exl3-bench/profile/d2s_*.py): 17.78 vs 16.62 t/s drafted with the full head; Kyojin's Yamz pack gave 17.72 (2026-10-21)
export EXL3_DRAFT2S="${EXL3_DRAFT2S-$P/models/flash-next/draft2s.st}"
eval "$(python3 "$(dirname "$0")/kyojin-env.py" "$TREE")"
export PYTHONPATH="$TREE"

cat > "$CFG" <<YML
network:
  host: 0.0.0.0
  port: $PORT
  disable_auth: false
  api_servers: ["OAI"]
  send_tracebacks: true
  sse_ping_interval: 15
logging:
  log_prompt: false
  log_generation_params: false
  log_requests: false
  log_chat_completion_requests: false
model:
  model_dir: $P/models
  model_name: flash-next
  backend: exllamav3
  inline_model_loading: false
  # froggeric/Qwen-Fixed-Chat-Templates v22.5, unmodified: stock Qwen put Swift1.5 agent first turns into "the user hasn't asked anything" ~70% of the time (2026-10-24). Defaults to medium effort; callers pass reasoning_effort
  prompt_template: $DEPLOY/flash-next.jinja
  max_seq_len: $CTX
  cache_size: $CTX
  # FP16 fits at 262K. 8,4 (2026-10-26, 1.6.5, 8K): -3.5 GiB GTT, decode flat, prefill 350 -> 247 t/s (quantized-cache staging)
  cache_mode: ${EXL3_CACHE_MODE:-FP16}
  chunk_size: 4096
  output_chunking: true
  max_batch_size: 1
  vision: true
  reasoning: true
  tool_format: qwen3_5
  start_in_reasoning: auto
  tool_calls_in_reasoning: true
draft_model:
  # EXL3_DRAFT_* (A/B only): e.g. a DFlash2 drafter is MODE=model NAME=<dir under tinyfiddler/models> POLICY=default
  draft_mode: ${EXL3_DRAFT_MODE:-mtp}
  draft_model_dir: /home/dbrain/dev/tinyfiddler/models
  draft_model_name: ${EXL3_DRAFT_NAME:-}
  draft_num_tokens: ${EXL3_DRAFT_TOKENS:-7}
  draft_policy: ${EXL3_DRAFT_POLICY:-tier}
  dynamic_draft: ${EXL3_DRAFT_DYNAMIC:-false}
  draft_confidence: ${EXL3_DRAFT_CONFIDENCE:-0.4}
  # exllamav3 1.6.5 sparse draft head columns (EXL3_SPARSE_HEAD, A/B only). With EXL3_DRAFT2S= and 4096 columns: 14.7 vs
  # 20.2 t/s drafted, acceptance 0.38 vs 0.72 (2026-10-26): the tier policy's depth rule reads draft_p, which only draft2s exports
  sparse_draft_head: ${EXL3_SPARSE_HEAD:-0}
sampling:
  temperature:
    override: 1.0
    force: false
  top_k:
    override: 20
    force: false
  top_p:
    override: 0.95
    force: false
  min_p:
    override: 0.0
    force: false
memory:
  cuda_malloc_async: false
  sysmem_kv_cache: 4096
  sysmem_recurrent_cache: 4096
  sysmem_multimodal_cache: 1024
  disk_cache_dir: /home/dbrain/dev/tinyfiddler/.exl3-diskcache-kyojin
  disk_cache_size: 131072
YML

cd "$P/tabbyAPI"
exec "$P/venv/bin/python" main.py --config "$CFG"
