# CarbonTeq vLLM fork ledger

## Status

**Published v0.25.1 maintenance candidate.** This fork carries one bounded
TurboQuant correction required by the CarbonTeq post-training runtime. The
source regression is validated locally; production qualification still
requires the locked runtime image and a real hybrid Qwen rollout.

Upstream repository: `https://github.com/vllm-project/vllm.git`

Upstream base and tag:
`752a3a504485790a2e8491cacbb35c137339ad34` (`v0.25.1`).

Expected remotes:

- `origin`: `git@github.com:carbonteq-ai/vllm.git`
- `upstream`: `https://github.com/vllm-project/vllm.git`

Published implementation commit:
`bd95cebbb6c7146f5b61bf39e532ca7591498430`.

## Maintained delta

### Preserve TurboQuant cache dtype during KV-cache reshape

vLLM 0.25.1 represents a TurboQuant cache with `TQFullAttentionSpec`, whose
`kv_quant_mode` remains `NONE`. The V1 GPU runner interpreted that marker as an
unquantized, skipped layer and replaced the selected `turboquant_*` preset with
`auto` while reshaping the cache. TurboQuant then rejected the unknown preset,
preventing hybrid attention and state-space models from starting.

`vllm/v1/worker/gpu_model_runner.py` now preserves the selected cache dtype for
`TQFullAttentionSpec` while retaining `auto` for genuinely unquantized layers.
The selection is isolated in `_get_layer_cache_dtype_str` so both branches are
covered directly by
`tests/v1/worker/test_gpu_model_runner.py`.

The guard backports the behavior proposed independently in upstream pull
requests [#48177](https://github.com/vllm-project/vllm/pull/48177) and
[#49798](https://github.com/vllm-project/vllm/pull/49798). The implementation
commit attributes both upstream authors. CarbonTeq is not opening a duplicate
upstream pull request.

## Compatibility and operating constraints

- The maintained line is based only on vLLM 0.25.1 and must be selected by full
  CarbonTeq fork commit.
- The runtime must keep PyTorch, CUDA compiler components, FlashInfer, and vLLM
  on the versions resolved by the post-training release lock.
- This correction preserves an explicitly selected `turboquant_*` preset. It
  does not add a new TurboQuant format, alter packed page sizes, or change
  per-token quantization behavior.
- `--kv-cache-dtype-skip-layers` continues to reshape ordinary
  `KVQuantMode.NONE` attention specs with dtype `auto`.
- CUDA-graph, eager-mode, MTP, and colocated training support remain separate
  qualification dimensions.

## Validation

Source checks:

```bash
ruff check \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/worker/test_gpu_model_runner.py
git diff --check
```

The two focused cache-dtype assertions pass against the compiled vLLM 0.25.1
extensions in the post-training Python 3.13 environment.

The release image must also run:

```bash
python -m pytest \
  tests/v1/worker/test_gpu_model_runner.py \
  -k "cache_dtype_is_preserved_when_reshaping or unquantized_cache_dtype_uses_auto_when_reshaping"
```

Production qualification requires a hybrid Qwen model using a
`turboquant_*` KV-cache preset to start, generate a non-empty response, and
retain its requested cache dtype. DAPO and SAMPO qualification must separately
cover their optimizer update, MTP counters when selected, and retained
observability evidence.

## Rebase and retirement

1. Fetch `upstream` and select a new immutable release base.
2. Check whether upstream has merged #48177, #49798, or equivalent behavior.
3. If the regression passes without the CarbonTeq commit, drop this delta.
4. Otherwise reapply the implementation commit, resolving
   `TQFullAttentionSpec` imports and `_reshape_kv_cache_tensors` as
   conflict-sensitive areas.
5. Run the focused source checks and production GPU qualification.
6. Update this ledger and the framework consumer page before advancing the
   immutable pin.

## Deferred behavior

- This fork does not claim general TurboQuant support across all models,
  attention backends, or CUDA-graph modes.
- It does not change CUDA toolkit discovery or FlashInfer JIT activation;
  those remain runtime-environment responsibilities.
- It does not qualify MTP together with TurboQuant. That combination requires
  explicit DAPO and SAMPO GPU evidence in the consuming framework.
