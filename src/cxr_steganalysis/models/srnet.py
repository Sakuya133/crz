"""Paper-structured SRNet adaptation for explicit comparison runs.

The grouping follows Figure 1 of Boroumand, Chen & Fridrich (TIFS 2019):
two type-1 groups, five unpooled type-2 residual groups, four downsampling
type-3 groups, one type-4 group, global average pooling, and two logits.

This is an independent PyTorch adaptation. Input scaling, initialization,
optimizer, and the NIH patch protocol are project choices, not a reproduction of
the authors' complete training recipe.
"""

from __future__ import annotations

import torch
from torch import nn


class Type1(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Type2(nn.Module):
    """Unpooled residual group; deliberately no activation after addition."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            Type1(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.body(inputs)


class Type3(nn.Module):
    """Residual group with 3x3 stride-2 average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            Type1(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs) + self.shortcut(inputs)


class SRNet(nn.Module):
    """SRNet architecture adaptation returning cover/stego logits."""

    def __init__(self) -> None:
        super().__init__()
        self.front = nn.Sequential(
            Type1(1, 64),
            Type1(64, 16),
            *(Type2(16) for _ in range(5)),
        )
        self.compact = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = nn.Sequential(
            Type1(256, 512),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(512, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"Expected input shape Bx1xHxW, got {tuple(inputs.shape)}")
        features = self.type4(self.compact(self.front(inputs))).flatten(1)
        return self.head(features)
