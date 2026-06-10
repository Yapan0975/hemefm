"""End-to-end multi-modal HemeFM model assembled from existing pieces.

Layout:
    RNA encoder        = HemeFMEncoder (pretrained, loaded from checkpoint)
    Mutation encoder   = MutationSetEncoder
    Methylation encoder = MethylationEncoder
    Fusion             = CrossAttnFusion (RNA-queries, mut + meth K/V)
    Task heads         = SubtypeHead, ELNRiskHead, SurvivalHead, DrugResponseHead

Forward returns a dict keyed by task name; values are head outputs (which
include 'loss' when the corresponding labels are supplied).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hemefm.models.fusion import CrossAttnFusion, MethylationEncoder, MutationSetEncoder
from hemefm.models.heads import DrugResponseHead, ELNRiskHead, SubtypeHead, SurvivalHead
from hemefm.models.transformer import HemeFMEncoder


@dataclass
class MultiModalConfig:
    n_subtypes: int = 8
    n_eln_classes: int = 3
    n_drugs: int = 122
    n_mutation_genes: int = 200          # AML driver-gene panel size
    n_variant_classes: int = 6
    n_methylation_features: int = 5000
    fusion_layers: int = 2
    fusion_heads: int = 8
    dropout: float = 0.1
    freeze_rna_encoder: bool = False
    unfreeze_after_epoch: int = 5


class MultiModalHemeFM(nn.Module):
    def __init__(
        self,
        rna_encoder: HemeFMEncoder,
        cfg: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MultiModalConfig()
        self.rna_encoder = rna_encoder
        d = rna_encoder.d_model

        self.mutation_encoder = MutationSetEncoder(
            gene_vocab_size=self.cfg.n_mutation_genes + 1,
            n_variant_classes=self.cfg.n_variant_classes,
            d_model=d,
            dropout=self.cfg.dropout,
        )
        self.methylation_encoder = MethylationEncoder(
            n_features=self.cfg.n_methylation_features,
            d_model=d,
            dropout=self.cfg.dropout,
        )
        self.fusion = CrossAttnFusion(
            d_model=d,
            n_heads=self.cfg.fusion_heads,
            dropout=self.cfg.dropout,
            n_layers=self.cfg.fusion_layers,
        )

        self.subtype_head = SubtypeHead(d, self.cfg.n_subtypes, dropout=self.cfg.dropout)
        self.eln_head = ELNRiskHead(d, n_classes=self.cfg.n_eln_classes, dropout=self.cfg.dropout)
        self.survival_head = SurvivalHead(d, dropout=self.cfg.dropout)
        self.drug_head = DrugResponseHead(d, n_drugs=self.cfg.n_drugs, dropout=self.cfg.dropout)

        if self.cfg.freeze_rna_encoder:
            for p in self.rna_encoder.parameters():
                p.requires_grad = False

    def unfreeze_rna_encoder(self) -> None:
        for p in self.rna_encoder.parameters():
            p.requires_grad = True

    # --------------------------------------------------------------- RNA path
    def _encode_rna(self, gene_ids: Tensor, bin_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return the per-token encoder hidden states (before the MFM head)."""
        x = self.rna_encoder.embed(gene_ids, bin_ids)
        return self.rna_encoder.encoder(x, src_key_padding_mask=attention_mask)

    # --------------------------------------------------------------- forward
    def forward(self, batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        rna_hidden = self._encode_rna(batch["gene_ids"], batch["bin_ids"], batch["attention_mask"])

        mut_tokens = None
        mut_mask = None
        if "mut_gene_ids" in batch:
            # Short-circuit: if mask is all-True (entire batch is padding), skip the mutation
            # encoder to avoid NaN propagation from softmax(-inf) inside its transformer.
            if "mut_attention_mask" in batch and torch.all(batch["mut_attention_mask"]):
                pass                                            # leave mut_tokens=None, mut_mask=None
            else:
                mut_tokens = self.mutation_encoder(
                    batch["mut_gene_ids"], batch["mut_variant_class_ids"], batch["mut_attention_mask"]
                )
                mut_mask = batch["mut_attention_mask"]

        meth_tokens = None
        if "methylation" in batch and not torch.all(torch.isnan(batch["methylation"])):
            meth_tokens = self.methylation_encoder(torch.nan_to_num(batch["methylation"], nan=0.0))

        fused = self.fusion(
            rna_tokens=rna_hidden,
            rna_attention_mask=batch["attention_mask"],
            mut_tokens=mut_tokens,
            mut_attention_mask=mut_mask,
            meth_tokens=meth_tokens,
        )
        cls_emb = fused[:, 0]                       # CLS position

        return {
            "subtype":  self.subtype_head(cls_emb, labels=batch.get("subtype_label")),
            "eln":      self.eln_head(cls_emb, labels=batch.get("eln_label")),
            "survival": self.survival_head(cls_emb, times=batch.get("os_time"), events=batch.get("os_event")),
            "drug":     self.drug_head(cls_emb, targets=batch.get("drug_response")),
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
