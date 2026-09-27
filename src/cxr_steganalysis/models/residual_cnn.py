"""Lightweight residual CNN baseline; this is intentionally not SRNet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FixedHighPass(nn.Conv2d):
    """Four fixed zero-sum filters used to expose local residual content."""

    def __init__(self) -> None:
        super().__init__(1, 4, kernel_size=5, padding=2, bias=False)
        kernels = torch.tensor([
            [
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, -1, 4, -1, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, -1, 0],
                [0, 0, 4, 0, 0],
                [0, -1, 0, -1, 0],
                [0, 0, 0, 0, 0],
            ],
        ], dtype=torch.float32).unsqueeze(1)
        scales = kernels.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        with torch.no_grad():
            self.weight.copy_(kernels / scales)
        self.weight.requires_grad_(False)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = F.silu(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return F.silu(output + residual)


class HighPassResidualCNN(nn.Module):
    """Small grayscale binary classifier operating on fixed high-pass residuals."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.high_pass = FixedHighPass()
        self.stem = nn.Sequential(
            nn.Conv2d(4, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_channels * 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected input shape Bx1xHxW, got {tuple(inputs.shape)}")
        residuals = torch.clamp(self.high_pass(inputs), -3.0, 3.0)
        features = self.blocks(self.stem(residuals))
        return self.head(self.pool(features).flatten(1)).squeeze(1)
