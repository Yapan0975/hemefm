"""Bidirectional Transformer encoder for rank-binned gene-token sequences.

Skeleton implementation matching the architecture specified in Manuscript §3.3-3.4
(B2 + B3 of the methodology blueprint). Production-time hyperparameters live in
`configs/model/hemefm_base.yaml`; this file is the shape-level reference.

Tokens are pairs (gene_id, expression_bin_id). The two streams are embedded
separately and summed — gene identity is the "position", expression bin is the
"value". Geneformer / scBERT use the same factorization.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class HemeFMEncoder(nn.Module):
    """Bidirectional Transformer encoder + masked-feature-modeling (MFM) head.

    Args:
        vocab_size: number of distinct gene identifiers in the dictionary.
        n_bins: number of rank-expression bins per sample.
        max_seq_len: maximum number of gene tokens per sample.
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of Transformer encoder layers.
        d_ff: feed-forward dimension (typically 4 * d_model).
        dropout: dropout for FFN + residual paths.
        attention_dropout: dropout inside scaled dot-product attention.
        mask_token_id: bin id used for [MASK]. Must be < n_bins.
        pad_token_id: gene id used for padding. Must be < vocab_size.
        cls_token_id: gene id used for [CLS]. Must be < vocab_size.
    """

    def __init__(
        self,
        vocab_size: int,
        n_bins: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        cls_token_id: int = 2,
        bin_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

        # `bin_vocab_size` accommodates the tokenizer convention where real bins
        # are offset by BIN_OFFSET=2 (0=PAD, 1=MASK), so the actual bin id range
        # is [2, n_bins + 2). Default to n_bins + 2 to support both conventions
        # — the lower bins (used by the legacy synthetic dataloader) remain valid.
        self.bin_vocab_size = bin_vocab_size if bin_vocab_size is not None else n_bins + 2

        self.gene_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.bin_emb = nn.Embedding(self.bin_vocab_size, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LN — more stable for deep stacks
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # MFM head — projects encoder output back to bin logits.
        # Output dim = bin_vocab_size so the tokenizer's offset bin ids (2..n_bins+1)
        # remain valid indices. Bins 0 (PAD) and 1 (MASK) are never targets in the
        # CE loss (filtered by labels==-100), so the first two output channels are
        # effectively unused — at most ~2 × d_model wasted parameters.
        self.mfm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.bin_vocab_size),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def embed(self, gene_ids: Tensor, bin_ids: Tensor) -> Tensor:
        """Sum the gene-identity and bin-value embeddings and normalize."""
        x = self.gene_emb(gene_ids) + self.bin_emb(bin_ids)
        return self.embed_dropout(self.embed_norm(x))

    def forward(
        self,
        gene_ids: Tensor,         # (B, L) long
        bin_ids: Tensor,          # (B, L) long
        attention_mask: Tensor,   # (B, L) bool — True for padded positions
    ) -> Tensor:
        """Encode a batch and return per-token bin logits.

        Returns:
            logits: (B, L, n_bins)
        """
        x = self.embed(gene_ids, bin_ids)
        # nn.TransformerEncoder expects ``src_key_padding_mask`` to be True at PAD.
        h = self.encoder(x, src_key_padding_mask=attention_mask)
        return self.mfm_head(h)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
