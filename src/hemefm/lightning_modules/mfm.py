"""Masked-Feature-Modeling LightningModule.

Wraps the HemeFMEncoder with cross-entropy loss over masked bin positions and
an AdamW + cosine-with-warmup optimizer. This is the pretraining task; downstream
multi-task fine-tuning will live in a separate module (Task #5).
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from hemefm.models import HemeFMEncoder


class MFMLightningModule(L.LightningModule):
    def __init__(
        self,
        encoder: HemeFMEncoder,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 100,
        max_steps: int = 1_000,
        min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        # ignore `encoder` itself so checkpoints stay portable across model configs
        self.save_hyperparameters(ignore=["encoder"])

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        gene_ids: Tensor,
        bin_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        return self.encoder(gene_ids, bin_ids, attention_mask)

    # ----------------------------------------------------------- shared step
    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        logits = self(batch["gene_ids"], batch["bin_ids"], batch["attention_mask"])
        # logits: (B, L, n_bins); labels: (B, L); -100 indicates not-masked.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )

        with torch.no_grad():
            mask = batch["labels"] != -100
            if mask.any():
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == batch["labels"][mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=loss.device)

        self.log(f"{stage}/mfm_loss", loss, prog_bar=True, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}/mfm_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    # ----------------------------------------------------------- optimizer
    def configure_optimizers(self) -> dict[str, object]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or name.endswith(".bias") or "LayerNorm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
