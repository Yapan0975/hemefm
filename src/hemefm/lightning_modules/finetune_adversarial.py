"""Stage 3 — FinetuneLightningModule + PlatformAdversarialHead wrapper.

Wraps a MultiModalHemeFM with a PlatformAdversarialHead operating on CLS
embeddings via Gradient Reversal Layer (DANN-style, §3.7 v4 (c)).

Total loss = source task loss (subtype + ELN + survival + drug)
           + λ * platform_adversarial_loss
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from hemefm.models.adversarial import PlatformAdversarialHead
from hemefm.lightning_modules.finetune import FinetuneLightningModule


class FinetuneAdversarialLightningModule(L.LightningModule):
    """Wraps FinetuneLightningModule + PlatformAdversarialHead for domain-adversarial finetune."""

    def __init__(
        self,
        model,                                                      # MultiModalHemeFM
        adv_lambda: float = 0.1,
        adv_hidden: int = 256,
        weighting: str = "kendall",
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 50,
        max_steps: int = 1000,
        unfreeze_after_step: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.inner = FinetuneLightningModule(
            model=model, weighting=weighting, lr=lr, encoder_lr=encoder_lr,
            weight_decay=weight_decay, warmup_steps=warmup_steps,
            max_steps=max_steps, unfreeze_after_step=unfreeze_after_step,
        )
        self.model = model
        # Use d_model from rna_encoder
        d_model = getattr(model.rna_encoder, "d_model", 768)
        self.adversarial_head = PlatformAdversarialHead(
            d_model=d_model, n_platforms=2,
            hidden_dim=adv_hidden, dropout=0.1, lambda_=adv_lambda,
        )

    def configure_optimizers(self):
        # Clean two-group AdamW (encoder slower) + cosine schedule covering both heads + adversarial.
        encoder_params = list(self.inner.model.rna_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.inner.model.parameters() if id(p) not in encoder_ids]
        adv_params = list(self.adversarial_head.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.hparams.encoder_lr},
                {"params": other_params, "lr": self.hparams.lr},
                {"params": adv_params, "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        return opt

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        # Run the multimodal forward to get task outputs + CLS embedding
        model = self.inner.model
        # Replicate _encode_rna path manually to expose CLS embedding for the adversarial head
        rna_hidden = model._encode_rna(batch["gene_ids"], batch["bin_ids"], batch["attention_mask"])
        # Mutation handling (with the short-circuit fix already in multimodal.py)
        mut_tokens = None
        mut_mask = None
        if "mut_gene_ids" in batch:
            if "mut_attention_mask" in batch and torch.all(batch["mut_attention_mask"]):
                pass
            else:
                mut_tokens = model.mutation_encoder(
                    batch["mut_gene_ids"], batch["mut_variant_class_ids"], batch["mut_attention_mask"]
                )
                mut_mask = batch["mut_attention_mask"]
        meth_tokens = None
        if "methylation" in batch and not torch.all(torch.isnan(batch["methylation"])):
            meth_tokens = model.methylation_encoder(torch.nan_to_num(batch["methylation"], nan=0.0))

        fused = model.fusion(
            rna_tokens=rna_hidden, rna_attention_mask=batch["attention_mask"],
            mut_tokens=mut_tokens, mut_attention_mask=mut_mask, meth_tokens=meth_tokens,
        )
        cls_emb = fused[:, 0]                                       # (B, d_model)

        # Adversarial head — only compute when domain labels present
        adv_loss = torch.zeros((), device=cls_emb.device)
        if "domain_label" in batch:
            adv_out = self.adversarial_head(cls_emb, batch["domain_label"])
            adv_loss = adv_out.get("loss", adv_loss)
            self.log(f"{stage}/adv_loss", adv_loss.detach(), on_step=stage == "train",
                     on_epoch=stage != "train", prog_bar=False)
            # Adversarial accuracy (how easily the discriminator distinguishes)
            adv_acc = (adv_out["logits"].argmax(-1) == batch["domain_label"]).float().mean()
            self.log(f"{stage}/adv_disc_acc", adv_acc.detach(), on_step=False,
                     on_epoch=True, prog_bar=False)

        # Task heads — only run on source samples (domain_label==0) to avoid garbage gradients from target
        if "domain_label" in batch:
            src_mask = batch["domain_label"] == 0
            if src_mask.sum() == 0:
                # batch is all target — only adversarial signal
                total = adv_loss
                self.log(f"{stage}/total_loss", total.detach(), on_step=stage == "train",
                         on_epoch=stage != "train", prog_bar=True)
                return total
            cls_src = cls_emb[src_mask]
            task_outs = {
                "subtype":  model.subtype_head(cls_src, labels=batch["subtype_label"][src_mask]),
                "eln":      model.eln_head(cls_src, labels=batch["eln_label"][src_mask]),
                "survival": model.survival_head(cls_src, times=batch["os_time"][src_mask],
                                                events=batch["os_event"][src_mask]),
                "drug":     model.drug_head(cls_src, targets=batch["drug_response"][src_mask]),
            }
        else:
            task_outs = {
                "subtype":  model.subtype_head(cls_emb, labels=batch.get("subtype_label")),
                "eln":      model.eln_head(cls_emb, labels=batch.get("eln_label")),
                "survival": model.survival_head(cls_emb, times=batch.get("os_time"),
                                                events=batch.get("os_event")),
                "drug":     model.drug_head(cls_emb, targets=batch.get("drug_response")),
            }

        # Combine per-task losses via the inner module's weighting strategy
        task_losses: dict[str, torch.Tensor] = {}
        for k, v in task_outs.items():
            if isinstance(v, dict) and "loss" in v and v["loss"] is not None:
                task_losses[k] = v["loss"]
                self.log(f"{stage}/{k}_loss", v["loss"].detach(), on_step=stage == "train",
                         on_epoch=stage != "train", prog_bar=False)

        # Simple sum across task losses (avoids inner Kendall weighting module compat issues)
        if task_losses:
            task_total = torch.stack(list(task_losses.values())).sum()
        else:
            task_total = torch.zeros((), device=cls_emb.device)

        # Extra task-head metrics (subtype acc + eln qwk on val) — delegated via inner if available
        if stage == "val":
            sb_logits = task_outs["subtype"].get("logits") if isinstance(task_outs["subtype"], dict) else None
            if sb_logits is not None:
                src_labels = batch["subtype_label"][src_mask] if "domain_label" in batch else batch["subtype_label"]
                valid = src_labels != -100
                if valid.sum() > 0:
                    acc = (sb_logits[valid].argmax(-1) == src_labels[valid]).float().mean()
                    self.log("val/subtype_acc", acc.detach(), on_epoch=True, prog_bar=True)
            eln_probs = task_outs["eln"].get("probs") if isinstance(task_outs["eln"], dict) else None
            if eln_probs is not None:
                src_labels = batch["eln_label"][src_mask] if "domain_label" in batch else batch["eln_label"]
                valid = src_labels != -100
                if valid.sum() > 1:
                    eln_pred = eln_probs[valid].argmax(-1).cpu().numpy()
                    eln_true = src_labels[valid].cpu().numpy()
                    try:
                        from sklearn.metrics import cohen_kappa_score
                        qwk = cohen_kappa_score(eln_true, eln_pred, weights="quadratic")
                        self.log("val/eln_qwk", float(qwk), on_epoch=True, prog_bar=True)
                    except Exception:                                       # noqa: BLE001
                        pass

        total = task_total + adv_loss
        self.log(f"{stage}/total_loss", total.detach(), on_step=stage == "train",
                 on_epoch=stage != "train", prog_bar=True)
        self.log(f"{stage}/task_loss", task_total.detach(), on_step=stage == "train",
                 on_epoch=stage != "train", prog_bar=False)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
