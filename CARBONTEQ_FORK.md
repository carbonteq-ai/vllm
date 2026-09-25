# CarbonTeq vLLM fork

Status: published development release.

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
- Published CarbonTeq functional commit:
  `28705a52e35688e152300de59f972f7fe56fcc12` (release `carbonteq-v0.29.1.dev3`;
  adds generic SM120 batch-invariant GEMM, split-KV attention, GDN chunk
  alignment, invariant CUDA RMSNorm and multi-turn prefix reuse to dev2 at
  `fbbba6698b2f8a912b94705cfc09eb4fd7243716`)

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
- A model-independent SM120 invariant-matmul rule (`_SM120_GENERIC_RULE`):
  60 cells keyed by output width N and row count M with BLOCK_K fixed at 64, so
  the reduction order never changes. It serves every shape the K2 table misses.
  The fp32 LM head (`head_dtype=float32`) goes through the same invariant
  kernel instead of cuBLAS.
- Batch invariance for hybrid layers: the short-conv backend declares support,
  and Qwen3.5 GDN layers use the Triton prefill, one decode kernel for pure and
  mixed steps, and one row per program in the gated RMSNorm. Other GDN
  families refuse invariant mode until validated.
- Batch-invariant split-KV in the Triton unified attention kernel: KV is cut
  into fixed segments at absolute positions (128 tokens, growing only with
  `max_model_len`) and folded in order, so the split and single-pass kernels
  return identical bits and the backend still picks between them by batch
  size. The sliding-window V mask moved into the V load, which lets head-512
  windowed layers fit SM120 shared memory.
- Invariant RMSNorm without a residual uses vLLM's CUDA kernel, whose block
  size is pinned under VLLM_BATCH_INVARIANT, whenever every row starts 16-byte
  aligned (so its scalar/vector read split, and reduction order, cannot vary);
  other layouts keep the Triton kernel. Strided q/k/v views are no longer
  copied first.
- Multi-turn prefix reuse for hybrid (Mamba/short-conv) and sliding-window
  models: once a request decodes, `KVCacheCoordinator.get_replay_boundaries`
  also makes its last computed block reachable, so the default sparse
  retention (`prefix_cache_retention_interval=0`, upstream since #52216)
  keeps the state a next turn needs. Before, only prompt boundaries were
  kept and every turn re-prefilled the previous turn's generated tokens:
  LFM2.5 AutomationBench replay at c16 prefilled 43% fewer tokens (486,770
  to 279,538 per collection) and recomputed 0.1% instead of 9.6% of its
  reusable context. Upstream candidate.

- LFM2 DSpark (candidate, `codex/lfm2-dspark`): `Lfm2ForCausalLM` implements the
  EAGLE-3 auxiliary hidden-state interface (embedding and each layer's output,
  `hidden + residual`), `Lfm2DSparkDraftModel` maps to `Qwen3DSparkModel` with
  its interleaved RoPE (`rope_is_neox_style`) and Markov-fed confidence head
  (SGLang's `markov_rank > 0` default), and LFM2/LFM2-MoE size their
  short-conv page padding with the speculative tokens, which the runtime
  `ShortConv` state already includes. Ported from SGLang #31041; the short-conv
  verify rollback it adds is covered by vLLM #50272 in this base. Evidence on the
  RTX PRO 6000 with `LiquidAI/LFM2.5-2.6B@654f9463` and
  `LiquidAI/LFM2.5-2.6B-DSpark@458cedab`, nine speculative tokens, greedy,
  256 output tokens (`tools/carbonteq/lfm2_dspark_check.py`): with
  `VLLM_BATCH_INVARIANT=1` DSpark output equals target-only output for all
  nine sequences, 2.31x at concurrency 1 and 2.11x at 32; without invariance
  2.10x and 1.27x; 2.24-2.34 of nine draft tokens accepted per step.
- DFlash and DSpark keep their trailing prefix-cache block
  (`SpeculativeConfig.use_eagle_block_drop` returns False for them, as vLLM
  #54163 and #57110 propose): their drafter KV at position i depends only on
  the target's state at i, so the EAGLE last-block drop only discarded exact
  cached blocks. Replaying 32 recorded LFM2.5 AutomationBench episodes per
  collection (16 tasks, temperature 0.8, rank-4 LoRA,
  `tools/carbonteq/lfm2_dspark_replay.sh`): the rollout binding without
  speculation takes 84-87 s per collection at 87.9% prefix hits; DSpark with
  the drop and the binding's 4 GiB KV budget 77 s at 14% (the drafter's KV
  shares the budget, cutting target capacity by 38%); with a 6.5 GiB budget
  60-64 s at 45%; with the drop removed 46 s at 88.0%, 1.85x faster.
  Batch-invariant greedy output still equals target-only output with prefix
  caching on. Nine speculative tokens beat five (49-52 s) and four (50-54 s)
  at this concurrency. A DSpark binding must size `kv_cache_memory_bytes` for
  target plus drafter (about 26 KiB per token for LFM2.5-2.6B).
- Session-aware prefix-cache eviction (candidate, `codex/lfm2-dspark`):
  `KVCacheManager` records, per `session_id`, the cached blocks its latest
  request left (`vllm/v1/core/kv_session_tracker.py`), and
  `release_session(session_id)` (`AsyncLLM`, `LLMEngine`, and
  `POST /v1/sessions/release`) moves the blocks no other live session holds
  to the front of the free queue. They stay cached until reused; a prompt
  shared with a sibling rollout keeps its LRU place. Replaying the DSpark
  collections above at the 4 GiB budget, releasing each episode's session
  when it ends takes 62.3-63.2 s per collection against 65.3-65.5 s with
  sessions tagged but not released (tracking itself costs nothing
  measurable), at 39.5-43.0% prefix hits either way. The hit rate is bounded
  by capacity, not eviction order: the 32 live episodes grow to about 250K
  tokens against 157K of cache, and the scheduler preempts 52-58 running
  requests per collection (none at 6.5 GiB). An FP8 drafter KV cache is not
  a capacity lever on this hybrid model: hybrid page-size unification pads
  the drafter group, so capacity rises only from 156,819 to 163,097 tokens
  (and FlashInfer's metadata builder then reads the target's `auto` cache
  dtype for the FP8 drafter group and fails to start).

## Compatibility constraints

- Tensor parallel size 1 and pipeline parallel size 1.
- Full-weight targets require at least two LoRA slots for fixed Uno plus startup
  profiling. Policy-LoRA targets require three slots and a supported rank at
  least as large as policy rank plus Uno rank.
- The target model vocabulary upper bound must be provided as
  `uno_mask_token_id`.
- Under batch invariance, models with GDN layers end partial prefill chunks
  on 64-token boundaries (FLA_CHUNK_SIZE), costing at most 63 tokens of step
  budget; the GDN chunk kernel is bit-exact only across aligned splits.
  Prefix caching with GDN is not yet validated under invariance (the fork
  warns at startup).
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
  tests/v1/spec_decode/test_lfm2_dspark.py \
  tests/v1/core/test_kv_session_release.py \
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

The multi-turn continuation boundary is covered by
`tests/v1/core/test_prefix_caching.py -k multi_turn_continuation` (the next
turn hits 0 tokens without it and 80 with it); the whole file passes (161).

Batch-invariance changes are covered by:

```bash
pytest -q \
  tests/v1/determinism/test_matmul_batch_invariant.py \
  tests/v1/determinism/test_attention_batch_invariant_segments.py \
  tests/v1/core/test_batch_invariant_prefill_split.py \
  tests/v1/determinism/test_rms_norm_cuda_batch_invariant.py \
  tests/kernels/attention/test_triton_unified_attention.py -k "not use_td"
```

On the RTX PRO 6000 these pass 45, 34 and 1,588 tests. End-to-end with
`VLLM_BATCH_INVARIANT=1`, logprobs are bit-exact at c1-c32 on K2-Horizon-7B,
LFM2.5-2.6B, Gemma-4-12B, Gemma-4-E4B, Qwen2.5-0.5B and Qwen3.5-2B/27B, and
across staggered arrivals on K2, LFM, both Gemma models and Qwen3.5-2B (with
chunked prefill on). At c4 the fork
runs 2.1-3.4x faster than the upstream invariant configuration (Gemma-4-12B
87 -> 215 tok/s, LFM2.5-2.6B 250 -> 850 tok/s).

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
