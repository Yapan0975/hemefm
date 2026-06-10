"""Drug-response head — joint prediction across the BeatAML 122-compound panel."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _pearson_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """1 − Pearson r across the masked (non-NaN) entries. Differentiable.

    Clamping is applied BEFORE the sqrt so the gradient of sqrt does not blow up
    when the variance is zero (constant predictions or constant targets).
    """
    if mask.sum() < 2:
        return pred.new_zeros(())
    p = pred[mask]
    t = target[mask]
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    num = (p_centered * t_centered).sum()
    den_sq = (p_centered.pow(2).sum() * t_centered.pow(2).sum()).clamp_min(1e-8)
    return 1.0 - num / torch.sqrt(den_sq)


class DrugResponseHead(nn.Module):
    """Predicts a vector of drug-response scores (e.g. AUC across drugs).

    `n_drugs` is the panel size (122 for BeatAML 2.0). Missing entries are
    handled by passing a boolean mask into the loss; NaN values in the target
    are auto-masked.
    """

    def __init__(self, d_model: int, n_drugs: int, dropout: float = 0.1, mse_weight: float = 0.5) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_drugs),
        )
        self.n_drugs = n_drugs
        self.mse_weight = mse_weight

    def forward(self, cls_emb: Tensor, targets: Tensor | None = None) -> dict[str, Tensor]:
        pred = self.proj(cls_emb)                         # (B, n_drugs)
        out = {"pred": pred}
        if targets is None:
            return out

        mask = ~torch.isnan(targets)
        target_filled = torch.where(mask, targets, torch.zeros_like(targets))

        mse_full = F.mse_loss(pred, target_filled, reduction="none")
        mse = (mse_full * mask).sum() / mask.sum().clamp_min(1.0)

        # Pearson loss aggregated per-drug then averaged
        pearson_terms = []
        for d in range(self.n_drugs):
            m = mask[:, d]
            if m.sum() >= 2:
                pearson_terms.append(_pearson_loss(pred[:, d], targets[:, d], m))
        pearson = torch.stack(pearson_terms).mean() if pearson_terms else pred.new_zeros(())

        loss = self.mse_weight * mse + (1.0 - self.mse_weight) * pearson
        out.update({"loss": loss, "mse": mse, "pearson_loss": pearson})
        return out
