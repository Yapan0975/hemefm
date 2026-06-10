"""ELN 2022 ordinal-risk head (favorable < intermediate < adverse)."""
from __future__ import annotations

from torch import Tensor, nn

from hemefm.losses.ordinal import CumulativeLinkOrdinal


class ELNRiskHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),                # single scalar logit
        )
        self.cumulative_link = CumulativeLinkOrdinal(n_classes=n_classes)
        self.n_classes = n_classes

    def forward(self, cls_emb: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        logit = self.proj(cls_emb).squeeze(-1)             # (B,)
        if labels is None:
            return {"logit": logit, "probs": self.cumulative_link.class_probs(logit)}
        loss, probs = self.cumulative_link(logit, labels)
        return {"loss": loss, "logit": logit, "probs": probs}
