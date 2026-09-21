"""Model shape and structure."""

from __future__ import annotations

import pytest
import torch

from dvit.model import ViTConfig, build_model


class TestModel:
    def test_output_shape(self) -> None:
        m = build_model()
        assert m(torch.randn(4, 3, 32, 32)).shape == (4, 10)

    def test_token_count_matches_patch_grid(self) -> None:
        cfg = ViTConfig(image_size=32, patch_size=4)
        assert cfg.num_patches == 64
        assert build_model().pos_embed.shape == (1, 65, 192)   # 64 patches + cls

    def test_drop_path_is_identity_in_eval(self) -> None:
        m = build_model({"drop_path": 0.9}).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert torch.equal(m(x), m(x))
