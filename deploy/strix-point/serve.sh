#!/bin/bash
# Flash-Next (Swift1.5 EXL3 4.05) on exllamav3 1.6.3 + Kyojin kernels (dbrain/exllamav3 strix-point) behind
# TabbyAPI (dbrain/tabbyAPI strix-point) on Strix Point (gfx1150, 96 GB). Started by exl3-flashnext.service via
# the drop-ins next to this file (install as /etc/systemd/system/<unit>.d/kyojin.conf); same port, API key and
# model id as the llama.cpp flashnext.service it replaces. usage: serve.sh [port]
set -euo pipefail
P=/home/dbrain/dev/tinyfiddler/exl3-prod
PORT="${1:-8765}"
CTX="${EXL3_CTX:-262144}"
CFG="$P/tabbyAPI/config.yml"

export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm EXL3_ROCM_SDK=/opt/rocm
# Without streaming the n-gram table is loaded into RAM and the box swaps
export EXL3_NGRAM_STREAM=1 EXL3_NGRAM_POPULATE=0
eval "$(python3 "$(dirname "$0")/kyojin-env.py" "$P/exllamav3")"
export PYTHONPATH="$P/exllamav3"

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
  max_seq_len: $CTX
  cache_size: $CTX
  # Q8 saves ~1 GiB per 131K tokens but costs ~4% decode (measured 2026-10-20); FP16 fits at 262K
  cache_mode: FP16
  chunk_size: 4096
  output_chunking: true
  max_batch_size: 1
  vision: true
  reasoning: true
  tool_format: qwen3_5
  start_in_reasoning: auto
  tool_calls_in_reasoning: true
draft_model:
  draft_mode: mtp
  draft_num_tokens: 7
  draft_policy: tier
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
