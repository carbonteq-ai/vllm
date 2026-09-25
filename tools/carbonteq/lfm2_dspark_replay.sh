#!/bin/sh
# Replay recorded LFM2.5 AutomationBench collections with the DSpark drafter
# (CarbonTeq; run from the agentic benchmark directory on the workstation).
set -eu
label="${1:-dspark-verify-dspark}"
tokens="${2:-9}"
spec=$(printf '{"speculative_config":{"method":"dspark","model":"LiquidAI/LFM2.5-2.6B-DSpark","revision":"458cedab07d0f7b2b05700c77e1aa463d43d6f04","num_speculative_tokens":%s%s}}' "$tokens" "${SPEC_EXTRA:-}")
if [ -n "${KV_BYTES:-}" ]; then
  spec=$(printf '%s' "$spec" | sed "s/}}\$/},\"kv_cache_memory_bytes\":${KV_BYTES}}/")
fi
if [ "${SPEC_METHOD:-dspark}" = ngram ]; then
  spec=$(printf '{"speculative_config":{"method":"ngram","num_speculative_tokens":%s,"prompt_lookup_max":4,"prompt_lookup_min":2}}' "$tokens")
fi
exec python replay.py --traces data/lfm26-sample-50.jsonl --model LiquidAI/LFM2.5-2.6B \
  --binding-config bindings/lfm25-rollout-c32-4k-v2.json --lora-dir lora-lfm25-r4 \
  --mode collections --groups-per-collection 16 --repeats 2 --temperature 0.8 \
  --label "$label" --output "results/$label.json" --override "$spec"
