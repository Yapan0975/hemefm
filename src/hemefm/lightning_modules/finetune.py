"""Multi-task fine-tuning LightningModule.

Wraps `MultiModalHemeFM` with Kendall uncertainty weighting (default) or
GradNorm (ablation) and exposes layer-wise unfreezing of the pretrained RNA
encoder. Each task's metric is logged separately so collapse can be diagnosed
from the W&B dashboard.
"""
from __future__ import annotations

import lightning as L
import torch
from torch import Tensor

from hemefm.losses import GradNorm, KendallUncertaintyWeighting, quadratic_weighted_kappa
from hemefm.models import MultiModalHemeFM

TASK_NAMES = ("subtype", "eln", "survival", "drug")


class FinetuneLightningModule(L.LightningModule):
    def __init__(
        self,
        model: MultiModalHemeFM,
        weighting: str = "kendall",                       # 'kendall' | 'gradnorm' | 'equal'
        gradnorm_alpha: float = 1.5,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,                         # smaller LR for pretrained encoder
        weight_decay: float = 0.05,
        warmup_steps: int = 100,
        max_steps: int = 10_000,
        min_lr_ratio: float = 0.1,
        unfreeze_after_step: int | None = None,           # None -> never unfreeze
    ) -> None:
        super().__init__()
        self.model = model
        self.weighting_mode = weighting
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.unfreeze_after_step = unfreeze_after_step
        self.save_hyperparameters(ignore=["model"])

        if weighting == "kendall":
            self.weighting = KendallUncertaintyWeighting(TASK_NAMES)
        elif weighting == "gradnorm":
            self.weighting = GradNorm(TASK_NAMES, alpha=gradnorm_alpha)
            self.automatic_optimization = False           # GradNorm needs custom backward
        elif weighting == "equal":
            self.weighting = None
        else:
            raise ValueError(f"unknown weighting: {weighting}")

    # --------------------------------------------------------------- step
    def _collect_losses(self, outputs: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
        return {name: outputs[name]["loss"] for name in TASK_NAMES if "loss" in outputs.get(name, {})}

    def _log_outputs(self, outputs: dict[str, dict[str, Tensor]], batch: dict[str, Tensor], stage: str) -> None:
        with torch.no_grad():
            if "subtype" in outputs and "subtype_label" in batch:
                acc = (outputs["subtype"]["logits"].argmax(dim=-1) == batch["subtype_label"]).float().mean()
                self.log(f"{stage}/subtype_acc", acc, prog_bar=True, on_epoch=True)
            if "eln" in outputs and "eln_label" in batch:
                pred = outputs["eln"]["probs"].argmax(dim=-1)
                qwk = quadratic_weighted_kappa(pred, batch["eln_label"], n_classes=3)
                self.log(f"{stage}/eln_qwk", qwk, prog_bar=True, on_epoch=True)
            if "drug" in outputs and "drug_response" in batch:
                pearson = 1.0 - outputs["drug"].get("pearson_loss", torch.tensor(1.0, device=self.device))
                self.log(f"{stage}/drug_pearson", pearson, on_epoch=True)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self.model(batch)
        losses = self._collect_losses(outputs)
        if not losses:
            zero = batch["gene_ids"].new_zeros((), dtype=torch.float32, requires_grad=True)
            return zero

        # Surface NaN/Inf per task BEFORE weighting, so the W&B trace identifies the offender.
        for name in list(losses):
            val = losses[name]
            if not torch.isfinite(val):
                if batch_idx == 0:
                    print(f"[warn] task {name!r} produced non-finite loss = {val.item()}; "
                          f"replacing with 0 for this step")
                losses[name] = val.new_zeros((), requires_grad=True)

        if isinstance(self.weighting, KendallUncertaintyWeighting):
            total, scaled = self.weighting(losses)
            for name, term in scaled.items():
                self.log(f"train/{name}_scaled", term.detach(), on_step=True)
        elif isinstance(self.weighting, GradNorm):
            total = self._gradnorm_step(losses)
        else:
            total = sum(losses.values())

        for name, loss in losses.items():
            self.log(f"train/{name}_loss", loss.detach(), on_step=True, on_epoch=True)
        self.log("train/total_loss", total.detach() if isinstance(total, Tensor) else total, prog_bar=True)
        self._log_outputs(outputs, batch, "train")

        # Optional layer-wise unfreeze
        if self.unfreeze_after_step is not None and self.global_step == self.unfreeze_after_step:
            print(f"[finetune] step={self.global_step}: unfreezing RNA encoder")
            self.model.unfreeze_rna_encoder()

        return total

    def _gradnorm_step(self, losses: dict[str, Tensor]) -> Tensor:
        opt, meta_opt = self.optimizers()
        opt.zero_grad()
        meta_opt.zero_grad()

        # shared parameter for gradient measurement (last RNA encoder layer's FFN)
        shared = next(reversed(list(self.model.rna_encoder.encoder.layers[-1].linear2.parameters())))

        total = self.weighting.weighted_total(losses)
        meta = self.weighting.update_weights(losses, shared_param=shared)

        self.manual_backward(total, retain_graph=True)
        meta.backward()
        opt.step()
        meta_opt.step()
        self.weighting.renormalize()
        return total

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        outputs = self.model(batch)
        losses = self._collect_losses(outputs)
        for name, loss in losses.items():
            self.log(f"val/{name}_loss", loss.detach(), on_epoch=True)
        if isinstance(self.weighting, KendallUncertaintyWeighting):
            total, _ = self.weighting(losses)
        else:
            total = sum(losses.values()) if losses else None
        if total is not None:
            self.log("val/total_loss", total.detach(), prog_bar=True, on_epoch=True)
        self._log_outputs(outputs, batch, "val")
        return outputs

    # --------------------------------------------------------------- opt
    def configure_optimizers(self) -> object:
        encoder_params, head_params, weight_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model.rna_encoder"):
                encoder_params.append(p)
            elif name.startswith("weighting"):
                weight_params.append(p)
            else:
                head_params.append(p)

        param_groups = [
            {"params": head_params, "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.encoder_lr, "weight_decay": self.weight_decay}
            )
        if isinstance(self.weighting, KendallUncertaintyWeighting):
            # Kendall log_sigma needs a much smaller LR than heads to avoid
            # large drift on noisy small-batch gradients. 0.1× of head LR is a
            # safe default (Kendall et al. 2018 use a similar order-of-magnitude).
            param_groups.append({"params": weight_params, "lr": self.lr * 0.1, "weight_decay": 0.0})

        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            import math
            progress = min((step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        if isinstance(self.weighting, GradNorm):
            meta_opt = torch.optim.Adam([self.weighting.weights], lr=self.lr * 10)
            return [opt, meta_opt], [{"scheduler": sch, "interval": "step"}]

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
