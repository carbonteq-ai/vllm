# CarbonTeq vLLM fork

Status: release candidate.

This branch carries CarbonTeq's native Uno Psi-Spec integration and optional
SM120 paged-attention backend. Posttrain may select this candidate for the
qualified inference and native-LoRA Uno profiles described below. Full-weight
and QLoRA Uno updates remain unsupported and must fail during Posttrain job
compilation.

## Upstream base

- Repository: `https://github.com/vllm-project/vllm`
- Commit: `44dd18fe0bb0f13157f97a5aa029b6604468fca6`
- Expected origin: `git@github.com:carbonteq-ai/vllm.git`
- Expected upstream: `https://github.com/vllm-project/vllm.git`
- Development branch: `codex/sm120-attention-platform`
- Published CarbonTeq commit: pending qualification and push

## Maintained delta

- `SpeculativeConfig(method="uno")` with a separately pinned adapter identity,
  noise policy, and vocabulary bound.
- One native `UnoProposer` sharing the target model and KV cache rather than
  allocating a second causal model.
- Position-gated LoRA application: the Uno adapter is active only on future
  noise rows; vLLM's target sampler, verifier, and logprob path remain
  authoritative.
- One complete eight-candidate Uno block: the clean target-policy root remains
  at candidate position zero followed by seven future Uno drafts. vLLM verifies
  the block in one target pass and may emit one correction token, for at most
  nine scheduler outputs per step.
- Native policy-LoRA composition: target and seed rows select the request
  policy adapter, while noise rows select an atomically refreshed
  policy-plus-Uno composite adapter.
- Compact proposal inputs containing the target-sampled seed plus noise rows.
- Eager Uno proposal forwards while target execution uses ordinary vLLM full
  decode graphs. Proposer replay is deliberately unavailable because it has
  not passed proposal/logprob equivalence under batch churn.
- An opt-in SM120 paged-KV attention backend extracted behind vLLM's attention
  contract. Target-model graph support is part of the backend capability;
  there is no environment-variable gate or silent backend substitution.
  The optional `sm120` extra selects `sm120-paged-attention` 0.1.0 at immutable
  commit `99a6fe0acbb4756735aa8e47236f8b74e3f7c4be`; its release wheel SHA-256 is
  `3149a539c3296afbc56dfec88e86012fa00c8ac167eb828fa95efe90fdd38430`.
- A deterministic SM120 prefill policy that freezes the qualified M64N64
  reduction tile. The direct fixed-Q/K/V oracle is bit-exact across c1-c8;
  full-model batch invariance still requires vLLM's invariant linear path.
- A dedicated SM120 invariant-matmul config family measured against K2's real
  transposed-weight layout. It keeps fixed K-reduction tiles, tuned c4-c32
  decode/proposal buckets, and the proven large-prefill configuration.

## Compatibility constraints

- Tensor parallel size 1 and pipeline parallel size 1.
- Full-weight targets require at least two LoRA slots for fixed Uno plus startup
  profiling. Policy-LoRA targets require three slots and a supported rank at
  least as large as policy rank plus Uno rank.
- The target model vocabulary upper bound must be provided as
  `uno_mask_token_id`.
- The clean root must not be emitted separately. Removing it from the verified
  block breaks autoregressive alignment and collapsed measured acceptance.
- Uno has no benchmark-only deterministic-noise or CUDA-graph switches. The
  production path uses stochastic uniform noise and an eager proposer.

## Regression tests

Run from the fork root in a complete vLLM development environment:

```bash
pytest -q \
  tests/config/test_uno_speculative_config.py \
  tests/v1/attention/test_sm120_fa4.py \
  tests/v1/spec_decode/test_uno.py \
  tests/lora/test_layers.py \
  tests/lora/test_lora_manager.py
ruff check \
  vllm/config/speculative.py \
  vllm/config/vllm.py \
  vllm/lora \
  vllm/v1/attention/backends/sm120_fa4.py \
  vllm/v1/spec_decode/uno.py \
  tests/config/test_uno_speculative_config.py \
  tests/v1/attention/test_sm120_fa4.py \
  tests/v1/spec_decode/test_uno.py
```

The retained RTX PRO environment passes the focused mapping tests and the
native GPU smokes recorded in the Posttrain consumer documentation. A real K2
rank-8 policy-LoRA optimizer step also passed across policy versions `0` and
`1`, with 32/32 finite target logprobs and maximum post-update logprob movement
`0.0655067` while completion tokens stayed stable.

The deterministic SM120 prefill policy also passes its focused CPU policy test
and an RTX PRO kernel oracle. The strict end-to-end checkpoint matched all
responses at c8, c16, and c32. Its original c4 mean was 270.45 output tok/s
versus 526.41 for the numerically unstable fast candidate; this localized the
remaining cost to invariant model linears.

The shape-tuned invariant-linear path supersedes that 270.45 tok/s checkpoint:
two matched c4 runs averaged 445.36 output tok/s, while two-repeat diagnostics
matched c8 8/8, c16 16/16, and c32 32/32 responses. The focused invariant
matmul suite passes 35 tests on SM120.

The release-clean production configuration removes the rejected prefix path,
proposer graph code, benchmark-only deterministic noise, and SM120 graph/debug
environment switches. With released `sm120-paged-attention` 0.1.0 it reached
435.78 output tok/s on the matched warm c4 5K-prompt/1K-output control, with
zero preemptions.

## Rebase procedure

1. Rebase this branch onto an explicitly selected immutable upstream commit.
2. Resolve speculative-config and GPU-runner seams without weakening the
   complete-block verification contract.
3. Run the focused suite, vLLM's relevant speculative/LoRA suites, and lint.
4. Repeat native target-logprob, tool-call, mixed-concurrency, lifecycle, and
   optimizer-update qualification on the RTX PRO.
5. Push the fork commit before advancing any Posttrain pin or lockfile.

## Qualified release boundary

- Native inference and native rank-8 policy-LoRA refresh are qualified.
- Full-weight Uno refresh is not qualified and is not part of this release.
- QLoRA Uno refresh is not qualified and is not part of this release.

## Stable-promotion gates

- Distributional equivalence against ordinary vLLM.
- Abort/drain and mixed-batch churn.
- Matched warm long-prompt throughput comparison.
- Complete block-width-eight oracle against ordinary vLLM, including
  output-budget/EOS boundaries and policy-LoRA refresh.
- Full-weight and QLoRA refresh require independent designs, live optimizer
  gates, and a later release before Posttrain may admit them.
