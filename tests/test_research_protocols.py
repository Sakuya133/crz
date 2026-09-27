import numpy as np
import pandas as pd
import pytest
import torch
from cxr_steganalysis.training.groupdro import GroupDRO, BalancedViewBatchSampler
from cxr_steganalysis.experiment import select_mixed_training, manifest_identity
from cxr_steganalysis.config import load_config


def test_groupdro_matches_exponential_update_and_resume():
    dro = GroupDRO(.1)
    loss = torch.tensor([1., 3., 4., 6.], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1])
    objective = dro(loss, groups)
    expected = torch.softmax(torch.tensor([.2, .5]), dim=0)
    torch.testing.assert_close(dro.log_q.exp(), expected)
    torch.testing.assert_close(objective, expected @ torch.tensor([2., 5.]))
    objective.backward()
    assert loss.grad is not None
    resumed = GroupDRO(.1)
    resumed.load_state_dict(dro.state_dict())
    torch.testing.assert_close(dro(loss, groups), resumed(loss, groups))
    assert dro.updates == resumed.updates == 2


def test_groupdro_large_losses_stable_and_missing_group_rejected():
    dro = GroupDRO(.1)
    assert torch.isfinite(dro(torch.tensor([1e6, 2e6]), torch.tensor([0, 1])))
    assert torch.isfinite(dro.log_q).all()
    with pytest.raises(ValueError):
        dro(torch.ones(2), torch.zeros(2, dtype=torch.long))


def test_balanced_sampler_no_drop_and_exact_resume():
    views = ["AP"]*350 + ["PA"]*350
    generator = torch.Generator().manual_seed(1337)
    sampler = BalancedViewBatchSampler(views, 8, generator)
    batches = list(sampler)
    assert len(batches) == 88 and len(batches[-1]) == 4
    assert sorted(sum(batches, [])) == list(range(700))
    for b in batches:
        assert sum(views[i] == "AP" for i in b)*2 == len(b)
    state = generator.get_state()
    expected = list(sampler)
    generator.set_state(state)
    assert list(sampler) == expected


def test_mixed_sampling_preserves_locked_split_and_test():
    frame = pd.DataFrame([dict(pair_id=f"{s}_{v}_{i}", patient_id=f"{s}{i}", split=s, view_position=v, cover_path="old") for s in ("train", "validation", "test") for v in ("AP", "PA") for i in range(6)])
    selected = select_mixed_training(frame, 3, 1337)
    assert selected[selected.split == "train"].groupby("view_position").size().tolist() == [3, 3]
    pd.testing.assert_frame_equal(frame[frame.split != "train"].reset_index(drop=True), selected[selected.split != "train"].reset_index(drop=True))
    assert manifest_identity(frame) == manifest_identity(frame.assign(cover_path="relocated"))
    with pytest.raises(ValueError):
        select_mixed_training(frame, 7, 1337)


def test_separate_protocol_configs():
    a = load_config("configs/pilot_existing.yaml")
    b = load_config("configs/pilot_mixed.yaml")
    assert a["protocol_id"] != b["protocol_id"]
    assert a["seeds"] == b["seeds"]
    assert a["paths"]["pair_manifest"] == b["paths"]["pair_manifest"]
