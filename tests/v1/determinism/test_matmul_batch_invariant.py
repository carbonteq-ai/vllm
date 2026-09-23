# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.determinism.batch_invariant import (
    matmul_batch_invariant,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    _get_tuned_matmul_arch_family,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

DEVICE_TYPE = current_platform.device_type


def test_sm120_uses_dedicated_tuned_config_family():
    assert _get_tuned_matmul_arch_family(DeviceCapability(12, 0)) == "sm120"


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """Compare matmul_batch_invariant against torch.matmul for various shapes."""
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@skip_unsupported
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["contiguous", "transposed"])
def test_matmul_batch_invariance_across_tuned_m_buckets(m, transpose_b):
    # Tuned M buckets must preserve each row's K-reduction order.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS:
        pytest.skip("No tuned persistent matmul config for this architecture")

    device = torch.device(DEVICE_TYPE)
    n = k = 2048
    torch.manual_seed(42)
    a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    if transpose_b:
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device).t()
    else:
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    single_output = matmul_batch_invariant(a[:1], b)
    batch_output = matmul_batch_invariant(a, b)

    assert torch.equal(single_output[0], batch_output[0])


def test_sm120_generic_rule_keeps_block_k_fixed():
    # The invariance contract: M buckets may change the tile, never BLOCK_K.
    from vllm.model_executor.determinism.batch_invariant_configs import (
        _SM120_GENERIC_BLOCK_K,
        _sm120_generic_matmul_config,
    )

    for n in (32, 96, 896, 2048, 3840, 15360, 43008, 262144):
        for k in (256, 2048, 5376, 21504):
            block_ks = {
                _sm120_generic_matmul_config(m, n, k)["BLOCK_SIZE_K"]
                for m in (1, 2, 4, 7, 16, 33, 36, 100, 144, 288, 512, 4096, 65536)
            }
            assert block_ks == {_SM120_GENERIC_BLOCK_K}


def test_sm120_table_shapes_keep_their_tuned_config(monkeypatch):
    import vllm.model_executor.determinism.batch_invariant_configs as configs

    monkeypatch.setattr(configs, "_TUNED_MATMUL_CONFIGS_RESOLVED", True)
    monkeypatch.setattr(configs, "_TUNED_MATMUL_ARCH_FAMILY", "sm120")
    monkeypatch.setattr(
        configs,
        "_TUNED_MATMUL_CONFIGS_FOR_DEVICE",
        configs._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["sm120"],
    )
    default = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}

    tuned = configs._get_matmul_config(36, 6144, 4096, torch.bfloat16, default)
    assert (tuned["BLOCK_SIZE_M"], tuned["BLOCK_SIZE_N"]) == (16, 128)

    # An unseen model's shape gets the generic rule, not the 128-row default.
    generic = configs._get_matmul_config(4, 8192, 3840, torch.bfloat16, default)
    assert generic == configs._sm120_generic_matmul_config(4, 8192, 3840)
    assert generic["BLOCK_SIZE_M"] < 128

    # Other dtypes keep the caller's default.
    assert configs._get_matmul_config(4, 8192, 3840, torch.float16, default) is default


def _is_sm120() -> bool:
    return (
        current_platform.is_cuda()
        and _get_tuned_matmul_arch_family(current_platform.get_device_capability())
        == "sm120"
    )


@skip_unsupported
@pytest.mark.skipif(not _is_sm120(), reason="SM120 generic rule")
@pytest.mark.parametrize(
    "n,k",
    [(96, 5120), (1152, 896), (8192, 3840), (21504, 2048), (43008, 5376)],
    ids=["tiny-n", "qwen0.5b-qkv", "gemma12b-qkv", "lfm-gate-up", "gemma31b-gate-up"],
)
def test_sm120_generic_rule_is_batch_invariant_across_buckets(n, k):
    from vllm.model_executor.determinism.batch_invariant import matmul_persistent

    torch.manual_seed(0)
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    rows = torch.randn((320, k), device="cuda").to(torch.bfloat16)
    reference = matmul_persistent(rows[:1], weight.t())[0]
    # Row counts straddle every decode bucket edge, including Uno c4/c32 blocks.
    for m in (2, 4, 5, 16, 17, 32, 33, 36, 64, 65, 144, 145, 288, 320):
        assert torch.equal(matmul_persistent(rows[:m], weight.t())[0], reference), m


@skip_unsupported
@pytest.mark.skipif(not _is_sm120(), reason="SM120 generic rule")
def test_fp32_output_head_is_batch_invariant():
    # An fp32 lm_head must not fall back to cuBLAS, whose kernel changes with M.
    from vllm.model_executor.determinism.batch_invariant import matmul_persistent

    torch.manual_seed(0)
    weight = (torch.randn((128000, 2048), device="cuda") * 0.02).to(torch.bfloat16)
    rows = torch.randn((64, 2048), device="cuda").to(torch.bfloat16)
    reference = matmul_persistent(rows[:1], weight.t(), out_dtype=torch.float32)[0]
    assert reference.dtype == torch.float32
    for m in (4, 16, 32, 64):
        out = matmul_persistent(rows[:m], weight.t(), out_dtype=torch.float32)
        assert torch.equal(out[0], reference), m
    torch.testing.assert_close(
        reference, (rows[:1].float() @ weight.float().t())[0], rtol=0, atol=2e-3
    )


def test_hybrid_backend_batch_invariance_declarations():
    # Declarations follow measured full-model logprob equality on SM120.
    # Short-conv (LFM2.5) is invariant. GDN is validated per layer family:
    # Qwen's layer is, with the Triton prefill kernel invariant mode selects;
    # families sharing the backend with different kernels are not.
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        KimiGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
        OlmoHybridGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend

    assert ShortConvAttentionBackend.supports_batch_invariance()
    assert GDNAttentionBackend.supports_batch_invariance()
    assert not GatedDeltaNetAttention.batch_invariance_validated
    assert QwenGatedDeltaNetAttention.batch_invariance_validated
    assert not KimiGatedDeltaNetAttention.batch_invariance_validated
    assert not OlmoHybridGatedDeltaNetAttention.batch_invariance_validated


def test_gdn_prefill_backend_is_triton_under_batch_invariance(monkeypatch):
    from types import SimpleNamespace

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen_gdn

    monkeypatch.setattr(qwen_gdn.envs, "VLLM_BATCH_INVARIANT", True)

    def config(backend: str):
        return SimpleNamespace(
            additional_config={"gdn_prefill_backend": backend},
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(linear_key_head_dim=128)
            ),
        )

    assert qwen_gdn._resolve_gdn_prefill_backend(config("auto"))[1] == "triton"
    assert qwen_gdn._resolve_gdn_prefill_backend(config("triton"))[1] == "triton"
    for backend in ("flashinfer", "cutedsl"):
        with pytest.raises(ValueError, match="not batch-invariant"):
            qwen_gdn._resolve_gdn_prefill_backend(config(backend))
