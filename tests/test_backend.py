"""Backend selection, including the refusals."""

from __future__ import annotations

import pytest
import torch

from dvit.dist import UnsupportedTopology, resolve_backend
from dvit.engine import amp_dtype_for_capability, pick_amp_dtype


class TestBackendResolution:
    def test_single_process_needs_no_backend(self) -> None:
        assert resolve_backend("mps", 1) is None

    def test_cuda_picks_nccl_and_cpu_picks_gloo(self) -> None:
        assert resolve_backend("cuda", 2) == "nccl"
        assert resolve_backend("cpu", 4) == "gloo"

    def test_mps_refuses_to_pretend_it_is_distributed(self) -> None:
        # The whole point: no silent fallback that reports fake multi device runs.
        with pytest.raises(UnsupportedTopology, match="MPS cannot participate"):
            resolve_backend("mps", 2)


class TestAmpSelection:
    def test_cpu_autocast_uses_bf16_without_a_scaler(self) -> None:
        dtype, needs_scaler = pick_amp_dtype(torch.device("cpu"))
        assert dtype is torch.bfloat16 and needs_scaler is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_pre_ampere_cards_get_fp16_plus_scaler(self) -> None:
        dtype, needs_scaler = pick_amp_dtype(torch.device("cuda"))
        assert needs_scaler == (dtype is torch.float16)


class TestAmpDtypeByCapability:
    """Regression guard for a bug found only on real hardware.

    On a Tesla T4, sm_75, `torch.cuda.is_bf16_supported()` returns True
    because recent PyTorch counts emulated bf16. Selecting bf16 there runs a
    software path and measures a slowdown while calling it mixed precision.
    Capability is the honest signal.
    """

    @pytest.mark.parametrize("major", [7, 6])
    def test_pre_ampere_gets_fp16_and_a_scaler(self, major: int) -> None:
        dtype, needs_scaler = amp_dtype_for_capability(major)
        assert dtype is torch.float16
        assert needs_scaler is True

    @pytest.mark.parametrize("major", [8, 9, 10])
    def test_ampere_and_later_get_bf16_without_a_scaler(self, major: int) -> None:
        dtype, needs_scaler = amp_dtype_for_capability(major)
        assert dtype is torch.bfloat16
        assert needs_scaler is False

    def test_turing_is_not_treated_as_bf16_capable(self) -> None:
        assert amp_dtype_for_capability(7)[0] is not torch.bfloat16
