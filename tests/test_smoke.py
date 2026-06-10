"""Smoke tests — run with `uv run pytest -q`."""
from __future__ import annotations

import torch

from hemefm.data import SyntheticRankBinDataModule
from hemefm.lightning_modules import MFMLightningModule
from hemefm.models import HemeFMEncoder


def _tiny_encoder() -> HemeFMEncoder:
    return HemeFMEncoder(
        vocab_size=200, n_bins=8, max_seq_len=32,
        d_model=32, n_heads=4, n_layers=2, d_ff=64,
    )


def test_encoder_forward_shape() -> None:
    enc = _tiny_encoder()
    gene_ids = torch.randint(3, 200, (2, 32))
    bin_ids = torch.randint(2, 8, (2, 32))
    attn = torch.zeros(2, 32, dtype=torch.bool)
    out = enc(gene_ids, bin_ids, attn)
    assert out.shape == (2, 32, 8), f"unexpected shape {out.shape}"


def test_mfm_loss_is_finite() -> None:
    enc = _tiny_encoder()
    lm = MFMLightningModule(encoder=enc, warmup_steps=2, max_steps=10)
    dm = SyntheticRankBinDataModule(
        n_train=8, n_val=4, seq_len=32, vocab_size=200, n_bins=8, batch_size=4,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    loss = lm._step(batch, "train")
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()    # gradients flow


def test_encoder_parameter_count_reasonable() -> None:
    enc = _tiny_encoder()
    n = enc.num_parameters()
    assert 1_000 < n < 200_000, f"unexpected parameter count: {n}"
