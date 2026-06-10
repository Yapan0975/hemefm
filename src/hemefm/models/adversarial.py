"""Stage 3 — Platform-adversarial domain adaptation head (§3.7 v4 (c)).

DANN-style (Ganin et al. 2015) unsupervised domain adaptation:
  - Source domain (BeatAML 2.0 RNA-seq, with ELN labels) → fine-tune the standard task heads.
  - Target domain (GSE6891 microarray, no labels) → also flows through encoder.
  - A platform-discriminator head reads the CLS embedding and tries to predict
    "is this sample from source or target?"
  - A Gradient Reversal Layer between encoder and discriminator multiplies
    gradients by −λ on the backward pass, so the encoder is encouraged to produce
    CLS embeddings that the discriminator CANNOT distinguish.
  - Net effect: encoder learns platform-invariant features at the representation
    level, while task heads still optimize for source-domain labels.

Pre-registered v4 §3.7 (c) (auxiliary loss weight 0.1 → λ=0.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Identity in forward, multiply gradient by -λ in backward."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):                   # noqa: D401
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 0.1) -> torch.Tensor:
    """Apply Gradient Reversal Layer (GRL) — identity forward, scale-negate gradients backward."""
    return _GradientReversalFunction.apply(x, lambda_)


class PlatformAdversarialHead(nn.Module):
    """Two-layer MLP that classifies the source platform of an input CLS embedding.

    Used adversarially via Gradient Reversal Layer — the encoder is encouraged
    to produce platform-invariant representations.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_platforms: int = 2,                                       # source (RNA-seq) vs target (microarray)
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lambda_: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_platforms),
        )

    def forward(self, cls_emb: torch.Tensor, platform_labels: torch.Tensor | None = None) -> dict:
        """Forward pass.

        Args:
            cls_emb: (B, d_model) — CLS embeddings from the encoder.
            platform_labels: (B,) int64 — 0 = source, 1 = target. None at inference.

        Returns:
            dict with `logits` (B, n_platforms), and `loss` (scalar, cross-entropy)
            if platform_labels provided.
        """
        # Gradient reversal between encoder and discriminator
        reversed_emb = gradient_reversal(cls_emb, self.lambda_)
        logits = self.mlp(reversed_emb)
        out = {"logits": logits}
        if platform_labels is not None:
            loss = nn.functional.cross_entropy(logits, platform_labels)
            out["loss"] = loss
        return out
