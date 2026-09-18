# CarbonTeq vLLM fork

This branch carries CarbonTeq's native Uno Psi-Spec integration.

Release candidate: `carbonteq-v0.26.1.dev1`. This is a source-overlay
prerelease for TRL and veRL qualification; it is not yet the stable Posttrain
production pin.

## Upstream base

- Repository: `https://github.com/vllm-project/vllm`
- Commit: `75c71390d5b399f5397a9166920fc45902f99f14`
- Development branch: `codex/uno-spec-decoding`
- Release branch: `codex/uno-spec-decoding`
- Candidate tag: `carbonteq-v0.26.1.dev1`
- Published CarbonTeq commit: the commit carrying this ledger update

## Maintained delta

- `SpeculativeConfig(method="uno")` with a separately pinned adapter identity,
  noise policy, and vocabulary bound.
- One native `UnoProposer` sharing the target model and KV cache rather than
  allocating a second causal model.
- Position-gated LoRA application: the Uno adapter is active only on future
  noise rows; vLLM's target sampler, verifier, and logprob path remain
  authoritative.
- Native policy-LoRA composition: target and seed rows select the request
  policy adapter, while noise rows select an atomically refreshed
  policy-plus-Uno composite adapter.
- Compact proposal inputs containing the target-sampled seed plus noise rows.
- Eager proposal forwards while target execution stays compiled. A dedicated,
  shape-safe Uno CUDA graph may replace this guard after qualification.

## Compatibility constraints

- Tensor parallel size 1 and pipeline parallel size 1.
- Full-weight targets require at least two LoRA slots for fixed Uno plus startup
  profiling. Policy-LoRA targets require three slots and a supported rank at
  least as large as policy rank plus Uno rank.
- The target model vocabulary upper bound must be provided as
  `uno_mask_token_id`.

## Regression tests

Run from the fork root in a complete vLLM development environment:

```bash
pytest -q \
  tests/config/test_uno_speculative_config.py \
  tests/v1/spec_decode/test_uno.py
ruff check \
  vllm/config/speculative.py \
  vllm/config/vllm.py \
  vllm/v1/spec_decode/uno.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/config/test_uno_speculative_config.py \
  tests/v1/spec_decode/test_uno.py
```

The retained RTX PRO environment passes all eight focused configuration and
proposer tests, plus the native GPU smokes recorded in the Posttrain consumer
documentation. A real K2
rank-8 policy-LoRA optimizer step also passed across policy versions `0` and
`1`, with 32/32 finite target logprobs and maximum post-update logprob movement
`0.0655067` while completion tokens stayed stable.

## Rebase procedure

1. Rebase this branch onto an explicitly selected immutable upstream commit.
2. Resolve speculative-config and GPU-runner seams without weakening the
   target-authoritative sampling contract.
3. Run the focused suite, vLLM's relevant speculative/LoRA suites, and lint.
4. Repeat native target-logprob, tool-call, mixed-concurrency, lifecycle, and
   optimizer-update qualification on the RTX PRO.
5. Push the fork commit before advancing any Posttrain pin or lockfile.

## Remaining stable-promotion gates

- Distributional equivalence against ordinary vLLM.
- Abort/drain and mixed-batch churn.
- Full-weight refresh across a known optimizer step with policy-version
  fencing and fresh target logprobs.
- Matched warm long-prompt throughput comparison.
- QLoRA remains independently unsupported.
