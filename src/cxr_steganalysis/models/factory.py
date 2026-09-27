"""Explicit model selection and checkpoint identity."""

from __future__ import annotations

from typing import Any

from torch import nn

from cxr_steganalysis.models.residual_cnn import HighPassResidualCNN
from cxr_steganalysis.models.srnet import SRNet


MODEL_VERSIONS = {
    "highpass": "HighPassResidualCNN-v1",
    "srnet": "SRNet-paper-structured-pytorch-v1",
    "hsmnet": "HSMNet-author-2c9f57be-adapter-v1",
}


def create_model(name: str, config: dict[str, Any]) -> nn.Module:
    model_name = str(name).lower()
    if model_name == "highpass":
        return HighPassResidualCNN(base_channels=int(config["model"]["base_channels"]))
    if model_name == "srnet":
        return SRNet()
    if model_name == "hsmnet":
        from cxr_steganalysis.models.hsmnet_adapter import create_hsmnet
        return create_hsmnet(config)
    raise ValueError(f"Unknown model {name!r}; choose highpass or srnet")


def model_identity(name: str) -> dict[str, str]:
    model_name = str(name).lower()
    if model_name not in MODEL_VERSIONS:
        raise ValueError(f"Unknown model {name!r}; choose highpass or srnet")
    return {"model_name": model_name, "model_version": MODEL_VERSIONS[model_name]}
