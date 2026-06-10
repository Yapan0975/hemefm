"""Cross-attention multi-modal fusion.

Inputs:
    rna_tokens:  (B, L_rna, D)  — required, from the pretrained HemeFMEncoder
    mut_tokens:  (B, L_mut, D)  — optional, from MutationSetEncoder
    meth_tokens: (B, L_meth, D) — optional, from MethylationEncoder

The fusion module consumes RNA tokens as queries and concatenates whatever
auxiliary modalities are present into a single key/value stream. Modality-type
embeddings are added before attention to preserve identity.

Missing modalities are handled by passing `None`; the module degrades to a
standard self-attention block over RNA tokens, ensuring the model remains
trainable on partial-modality samples (~25 % of BeatAML lacks methylation).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MutationSetEncoder(nn.Module):
    """Set-Transformer-style encoder over a variable-length list of mutations.

    Each mutation is represented as a (gene_id, variant_class_id) pair where
    gene_id picks a row in the pretrained gene-embedding table (shared with the
    RNA encoder if `share_gene_embedding=True`) and variant_class_id picks a
    class from {missense, nonsense, frameshift, splice, in-frame indel, other}.
    """

    def __init__(
        self,
        gene_vocab_size: int,
        n_variant_classes: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        shared_gene_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if shared_gene_embedding is not None:
            self.gene_emb = shared_gene_embedding
            if self.gene_emb.num_embeddings < gene_vocab_size:
                raise ValueError("shared gene embedding too small for mutation vocab")
        else:
            self.gene_emb = nn.Embedding(gene_vocab_size, d_model, padding_idx=0)
        self.variant_emb = nn.Embedding(n_variant_classes + 1, d_model)   # +1 for PAD
        self.norm_in = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

    def forward(self, gene_ids: Tensor, variant_class_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.gene_emb(gene_ids) + self.variant_emb(variant_class_ids)
        x = self.norm_in(x)
        return self.encoder(x, src_key_padding_mask=attention_mask)


class MethylationEncoder(nn.Module):
    """MLP over per-gene aggregated β-values (5,000 most-variable CpG-gene blocks).

    Returns a single (B, 1, D) token suitable for cross-attention.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, beta: Tensor) -> Tensor:                # (B, n_features) -> (B, 1, D)
        return self.proj(beta).unsqueeze(1)


class CrossAttnFusion(nn.Module):
    """Cross-attention block fusing RNA queries with mut + meth keys/values."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, n_layers: int = 2) -> None:
        super().__init__()
        self.modality_emb = nn.Embedding(3, d_model)   # 0=RNA, 1=mut, 2=meth
        self.layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        rna_tokens: Tensor,                         # (B, L_rna, D)
        rna_attention_mask: Tensor,                 # (B, L_rna) — True = pad
        mut_tokens: Tensor | None = None,           # (B, L_mut, D)
        mut_attention_mask: Tensor | None = None,
        meth_tokens: Tensor | None = None,          # (B, L_meth, D)
        meth_attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Build the auxiliary KV stream if any modality is provided.
        aux_tokens: list[Tensor] = []
        aux_masks: list[Tensor] = []
        if mut_tokens is not None:
            mut_with_id = mut_tokens + self.modality_emb.weight[1].view(1, 1, -1)
            aux_tokens.append(mut_with_id)
            if mut_attention_mask is None:
                mut_attention_mask = torch.zeros(mut_tokens.shape[:-1], dtype=torch.bool, device=mut_tokens.device)
            aux_masks.append(mut_attention_mask)
        if meth_tokens is not None:
            meth_with_id = meth_tokens + self.modality_emb.weight[2].view(1, 1, -1)
            aux_tokens.append(meth_with_id)
            if meth_attention_mask is None:
                meth_attention_mask = torch.zeros(meth_tokens.shape[:-1], dtype=torch.bool, device=meth_tokens.device)
            aux_masks.append(meth_attention_mask)

        if aux_tokens:
            kv = torch.cat(aux_tokens, dim=1)
            kv_mask = torch.cat(aux_masks, dim=1)
        else:
            kv = rna_tokens
            kv_mask = rna_attention_mask

        h = rna_tokens
        for block in self.layers:
            h = block(h, kv, kv_mask)
        return self.norm_out(h)


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, kv: Tensor, kv_mask: Tensor) -> Tensor:
        h = self.norm_self(x)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)
        h = self.norm_cross(x)
        a, _ = self.cross_attn(h, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x
