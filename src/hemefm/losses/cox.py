"""Cox partial-likelihood loss with Efron tie correction.

Reference:
    Katzman JL et al. (2018) DeepSurv: Personalized treatment recommender
    system using a Cox proportional hazards deep neural network.
    BMC Med Res Methodol 18:24. DOI 10.1186/s12874-018-0482-1

Numerically stable implementation:
    NPLL = - Σ_i δ_i [ θ_i - log Σ_{j ∈ R(t_i)} exp(θ_j) ]
where δ_i = event indicator, θ_i = log-risk score, R(t_i) = risk set at time t_i.

Efron correction for tied event times: when k events tie at the same time,
each event's contribution to the denominator uses a fractional sum of the
tied risks. We compute it in fp32 even under bf16 autocast to avoid NaNs.
"""
from __future__ import annotations

import torch
from torch import Tensor


def cox_partial_likelihood(
    log_risk: Tensor,                  # (B,) log-risk scores from the survival head
    times: Tensor,                     # (B,) follow-up times
    events: Tensor,                    # (B,) event indicators (1 = event, 0 = censored)
    *,
    tie_correction: str = "efron",
    eps: float = 1e-7,
) -> Tensor:
    """Negative partial log-likelihood; lower is better. Returns scalar (mean over events)."""
    if tie_correction not in ("breslow", "efron"):
        raise ValueError(f"unknown tie_correction: {tie_correction}")

    with torch.autocast(device_type=log_risk.device.type, enabled=False):
        log_risk = log_risk.float()
        times = times.float()
        events = events.float()

        # Sort by descending time so prefix logcumsumexp(θ) is log Σ_{j ∈ R(t)} exp(θ_j).
        order = torch.argsort(times, descending=True)
        log_risk_s = log_risk[order]
        times_s = times[order]
        events_s = events[order]

        # log Σ exp via torch.logcumsumexp — numerically stable.
        log_cum = torch.logcumsumexp(log_risk_s, dim=0)        # (B,)

        if tie_correction == "breslow":
            ll = events_s * (log_risk_s - log_cum)
            return -ll.sum() / events_s.sum().clamp_min(1.0)

        # ---- Efron correction --------------------------------------------
        # For each unique event time with k tied events, the correction is:
        #   contribution = Σ_{j∈ties} θ_j − Σ_{r=0..k-1} log(S − r/k · T)
        # where S = Σ_{j∈R(t)} exp(θ_j) (full risk set incl. ties) and
        #       T = Σ_{j∈ties∩events} exp(θ_j).
        # We compute log(S − r/k · T) via log-space subtraction:
        #   log(S − r/k · T) = log(S) + log(1 − (r/k) · exp(log T − log S))
        ll_total = log_risk_s.new_zeros(())
        n_events = events_s.sum().clamp_min(1.0)

        i = 0
        n = times_s.numel()
        while i < n:
            j = i
            while j + 1 < n and times_s[j + 1] == times_s[i]:
                j += 1
            tied_event_mask = events_s[i:j + 1] == 1
            k = int(tied_event_mask.sum().item())
            if k > 0:
                # log T and log S in log-space.
                tied_event_log_risks = log_risk_s[i:j + 1][tied_event_mask]
                log_T = torch.logsumexp(tied_event_log_risks, dim=0)
                log_S = log_cum[j]
                tied_lr_sum = tied_event_log_risks.sum()

                # ratio = T / S = exp(log_T - log_S). Clamp into (0, 1) just in case.
                ratio = torch.exp(log_T - log_S).clamp(0.0, 1.0)

                for r in range(k):
                    # log(S − (r/k) · T) = log_S + log1p(-(r/k) · ratio)
                    inside = (-((r / k) * ratio)).clamp_min(-1.0 + 1e-7)
                    ll_total = ll_total + log_S + torch.log1p(inside)
                ll_total = ll_total - tied_lr_sum
            i = j + 1

        return ll_total / n_events
