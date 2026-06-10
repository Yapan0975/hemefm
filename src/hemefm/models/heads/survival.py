"""Survival head — outputs a log-risk score for Cox partial likelihood."""
from __future__ import annotations

from torch import Tensor, nn

from hemefm.losses.cox import cox_partial_likelihood


class SurvivalHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        cls_emb: Tensor,
        times: Tensor | None = None,
        events: Tensor | None = None,
        tie_correction: str = "efron",
    ) -> dict[str, Tensor]:
        log_risk = self.proj(cls_emb).squeeze(-1)
        out = {"log_risk": log_risk}
        if times is not None and events is not None:
            out["loss"] = cox_partial_likelihood(log_risk, times, events, tie_correction=tie_correction)
        return out
