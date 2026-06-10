"""Cumulative-link ordinal regression for ELN 2022 (favorable < intermediate < adverse).

Model output: scalar logit per sample. Class boundaries θ_0 < θ_1 are learnable.
P(y ≤ k | x) = sigmoid(θ_k - logit). Class probabilities are differences of CDFs.

Loss: negative log-likelihood on the observed class.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CumulativeLinkOrdinal(nn.Module):
    """Proportional-odds head with K classes and K-1 learnable thresholds."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("ordinal head needs at least 2 classes")
        self.n_classes = n_classes
        # Initialize thresholds to spread roughly evenly along the logit axis.
        init = torch.linspace(-1.0, 1.0, n_classes - 1)
        self.thresholds = nn.Parameter(init)

    def class_probs(self, logits: Tensor) -> Tensor:
        """Return (B, K) class probabilities from (B,) logits."""
        ordered = torch.sort(self.thresholds)[0]
        # CDFs at each threshold:  (B, K-1)
        cdf = torch.sigmoid(ordered.unsqueeze(0) - logits.unsqueeze(1))
        ones = torch.ones_like(cdf[:, :1])
        zeros = torch.zeros_like(cdf[:, :1])
        cdf_with_bounds = torch.cat([zeros, cdf, ones], dim=1)
        probs = cdf_with_bounds[:, 1:] - cdf_with_bounds[:, :-1]
        return probs.clamp_min(1e-8)

    def forward(self, logits: Tensor, labels: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Return class_probs; if labels given, also return NLL."""
        probs = self.class_probs(logits)
        if labels is None:
            return probs
        nll = F.nll_loss(probs.log(), labels.long())
        return nll, probs


def quadratic_weighted_kappa(preds: Tensor, labels: Tensor, n_classes: int) -> Tensor:
    """Differentiable QWK proxy (computed from predicted class via argmax)."""
    preds = preds.long()
    labels = labels.long()
    O = torch.zeros(n_classes, n_classes, device=preds.device, dtype=torch.float32)
    for p, t in zip(preds.tolist(), labels.tolist()):
        O[t, p] += 1
    hist_t = O.sum(dim=1)
    hist_p = O.sum(dim=0)
    E = torch.outer(hist_t, hist_p) / max(O.sum().item(), 1.0)
    idx = torch.arange(n_classes, device=preds.device, dtype=torch.float32)
    W = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2) / ((n_classes - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum().clamp_min(1e-7)
    return 1.0 - num / den
