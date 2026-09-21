"""A small vision transformer sized for 32x32 CIFAR-10.

Deliberately written out rather than imported from timm. The point of the
project is the training system, but a reviewer will ask whether you understand
what you are distributing, and a 120 line ViT answers that in the interview.

Default config is 1,806,538 parameters: patch 4 gives 64 tokens, which keeps
attention cheap enough that the model is not the bottleneck on a T4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    drop_path: float = 0.1

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


class DropPath(nn.Module):
    """Stochastic depth on the residual branch."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(
            cfg.dim, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.dim),
            nn.Dropout(cfg.dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ViTConfig()
        c = self.cfg

        self.patch_embed = nn.Conv2d(
            c.in_channels, c.dim, kernel_size=c.patch_size, stride=c.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, c.num_patches + 1, c.dim))
        self.pos_drop = nn.Dropout(c.dropout)

        rates = torch.linspace(0, c.drop_path, c.depth).tolist()
        self.blocks = nn.ModuleList([Block(c, r) for r in rates])
        self.norm = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(overrides: dict | None = None) -> ViT:
    cfg = ViTConfig(**(overrides or {}))
    return ViT(cfg)
