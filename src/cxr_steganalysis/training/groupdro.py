"""Two-view GroupDRO adaptation of Sagawa et al. (ICLR 2020).

Unadjusted, unnormalized exponential adversary update, evaluated in log space.
Group annotations enter only the objective, never the classifier input.
"""
import torch
from torch import nn
from torch.utils.data import Sampler


class GroupDRO(nn.Module):
    def __init__(self, step_size: float = .01):
        super().__init__()
        if step_size <= 0:
            raise ValueError("GroupDRO step size must be positive")
        self.step_size = float(step_size)
        self.register_buffer("log_q", torch.full((2,), -torch.log(torch.tensor(2.)).item()))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    def forward(self, losses, groups):
        if set(groups.tolist()) != {0, 1}:
            raise ValueError("Controlled GroupDRO requires both AP and PA in every batch")
        group_loss = torch.stack([losses[groups == g].float().mean() for g in (0, 1)])
        if not torch.isfinite(group_loss).all():
            raise FloatingPointError("Non-finite group losses; adversary state not updated")
        with torch.no_grad():
            new_log_q = self.log_q + self.step_size * group_loss.detach()
            self.log_q.copy_(new_log_q - torch.logsumexp(new_log_q, dim=0))
            self.updates.add_(1)
        return (self.log_q.exp().detach() * group_loss).sum()


class BalancedViewBatchSampler(Sampler):
    """Without replacement; equal views per batch, retains the final partial batch."""
    def __init__(self, views, batch_size, generator):
        self.indices = [[i for i, v in enumerate(views) if v == name] for name in ("AP", "PA")]
        if sum(map(len, self.indices)) != len(views) or len(self.indices[0]) != len(self.indices[1]):
            raise ValueError("Balanced sampler requires equal AP/PA counts and no other views")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Mixed-view batch size must be positive and even")
        self.half = batch_size // 2
        self.generator = generator

    def __len__(self):
        return (len(self.indices[0]) + self.half - 1) // self.half

    def __iter__(self):
        shuffled = [[idx[i] for i in torch.randperm(len(idx), generator=self.generator).tolist()] for idx in self.indices]
        for start in range(0, len(shuffled[0]), self.half):
            yield shuffled[0][start:start+self.half] + shuffled[1][start:start+self.half]
