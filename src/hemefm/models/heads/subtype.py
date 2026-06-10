"""Subtype classification head (WHO 2022 / ICC 2022 / FAB-derived classes)."""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SubtypeHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, cls_emb: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        logits = self.proj(cls_emb)
        out = {"logits": logits, "probs": logits.softmax(dim=-1)}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out
