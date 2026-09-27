from __future__ import annotations

import torch

from cxr_steganalysis.models.factory import create_model
from cxr_steganalysis.models.residual_cnn import HighPassResidualCNN
from cxr_steganalysis.models.srnet import SRNet, Type2, Type3


def test_model_forward_and_fixed_high_pass():
    model = HighPassResidualCNN(base_channels=8)
    logits = model(torch.rand(3, 1, 32, 32))
    assert logits.shape == (3,)
    assert model.high_pass.weight.requires_grad is False
    assert all(
        parameter.requires_grad is False
        for parameter in model.high_pass.parameters()
    )


def test_srnet_group_counts_and_output_shape():
    model = SRNet()
    output = model(torch.randn(2, 1, 32, 32))
    assert output.shape == (2, 2)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert sum(isinstance(module, Type2) for module in model.modules()) == 5
    assert sum(isinstance(module, Type3) for module in model.modules()) == 4
    assert create_model("srnet", {"model": {}}).__class__ is SRNet
