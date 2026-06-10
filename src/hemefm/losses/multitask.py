"""Multi-task loss weighting strategies.

Two strategies implemented:

1. Kendall homoscedastic uncertainty (CVPR 2018):
       total = Σ_k (1/(2σ_k²)) L_k + log σ_k
   Each task learns a log_sigma. Loss scales with task uncertainty.

2. GradNorm (ICML 2018):
       Maintains per-task weights w_k. After each step, computes the L2 norm
       of the gradient of (w_k * L_k) at the shared encoder's final layer,
       compares to the average rate of decrease, and updates w_k to balance
       gradient magnitudes. Used as ablation arm in the manuscript.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class KendallUncertaintyWeighting(nn.Module):
    """Learns log_sigma per task. Total loss = Σ_k (1/(2σ_k²)) L_k + log σ_k.

    Initializing log_sigma at 0 starts every task with weight 0.5 and zero log-term.
    """

    def __init__(self, task_names: Iterable[str]) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.log_sigma = nn.Parameter(torch.zeros(len(self.task_names)))

    LOG_SIGMA_MIN = -3.0
    LOG_SIGMA_MAX = 3.0

    def forward(self, losses: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        total = losses[self.task_names[0]].new_zeros(())
        scaled: dict[str, Tensor] = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            # Clamp log_sigma to keep `exp(-2*log_sigma)` from overflowing under
            # noisy small-batch gradients. Bounds chosen so weight ∈ [≈0.001, ≈200].
            log_sigma = self.log_sigma[i].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)
            # Stable form: term = 0.5 * exp(-2*log_sigma) * L + log_sigma
            weight = 0.5 * torch.exp(-2.0 * log_sigma)
            term = weight * losses[name] + log_sigma
            scaled[name] = term
            total = total + term
        return total, scaled

    def weights_summary(self) -> dict[str, float]:
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma)
            return {n: 0.5 / (sigma[i].item() ** 2) for i, n in enumerate(self.task_names)}


class GradNorm(nn.Module):
    """Gradient-magnitude balancing.

    Args:
        task_names: task identifier list.
        alpha: asymmetry hyperparameter (Chen 2018). 0 = equalize gradient magnitudes;
               larger values give faster-learning tasks lower weights.
        shared_param: tensor whose ‖∇ L_k‖ is used for measurement. Typically the
               last shared encoder weight.
    """

    def __init__(self, task_names: Iterable[str], alpha: float = 1.5) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(len(self.task_names)))
        self._initial_losses: dict[str, float] = {}

    def record_initial_losses(self, losses: dict[str, Tensor]) -> None:
        for n in self.task_names:
            if n in losses and n not in self._initial_losses:
                self._initial_losses[n] = float(losses[n].detach())

    def weighted_total(self, losses: dict[str, Tensor]) -> Tensor:
        total = losses[self.task_names[0]].new_zeros(())
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            total = total + self.weights[i].abs() * losses[name]
        return total

    def update_weights(
        self,
        losses: dict[str, Tensor],
        shared_param: Tensor,
    ) -> Tensor:
        """Return the GradNorm meta-loss (caller backprops it through `self.weights`)."""
        self.record_initial_losses(losses)

        gnorms: dict[str, Tensor] = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            grad = torch.autograd.grad(
                self.weights[i].abs() * losses[name],
                shared_param,
                retain_graph=True,
                create_graph=True,
            )[0]
            gnorms[name] = grad.norm(p=2)

        if not gnorms:
            return shared_param.new_zeros(())

        # Inverse training rate L_k(t)/L_k(0)
        rates: dict[str, Tensor] = {
            n: losses[n].detach() / max(self._initial_losses.get(n, 1.0), 1e-7)
            for n in gnorms
        }
        mean_rate = torch.stack(list(rates.values())).mean()
        relative = {n: rates[n] / mean_rate for n in rates}

        mean_gnorm = torch.stack(list(gnorms.values())).mean().detach()
        targets = {n: mean_gnorm * (relative[n] ** self.alpha) for n in gnorms}

        meta = sum(F.l1_loss(gnorms[n], targets[n].detach()) for n in gnorms)
        return meta

    def renormalize(self) -> None:
        """Keep Σ |w_k| = N task count, to prevent drift."""
        with torch.no_grad():
            w = self.weights.abs()
            self.weights.copy_(w * len(self.task_names) / w.sum().clamp_min(1e-7))
