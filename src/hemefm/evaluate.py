"""External-cohort inference + metric computation for the HemeFM multimodal model.

Loads a finetuned MultiModalHemeFM checkpoint and runs inference on any cohort
already present in `pretrain_corpus.parquet` (per `cohort` field of the metadata
parquet). Outputs:

    outputs/eval/<cohort>/predictions.tsv     — sample_id, ELN_pred, subtype_pred, OS_pred, drug_<name>_pred...
    outputs/eval/<cohort>/embeddings.parquet  — CLS-token embeddings (768-dim float32)
    outputs/eval/<cohort>/metrics.json        — paired metrics (if labels are loaded)

Usage:
    python -m hemefm.evaluate \\
        --ckpt logs/finetune-v3-beataml707-3gpu/version_0/checkpoints/last.ckpt \\
        --cohort TCGA-LAML \\
        --labels-tsv data/external/tcga_laml_labels.tsv          # optional
        --output-dir outputs/eval/tcga_laml

Compatible with TCGA-LAML, TARGET-AML, GSE6891, GSE37642, and the held-out 20-
sample TCGA partition (must already be tagged with cohort="TCGA-LAML-test").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Optional metric deps (installed by default in our venv)
try:
    from sklearn.metrics import cohen_kappa_score, roc_auc_score, accuracy_score
except ImportError:                                                 # pragma: no cover
    cohen_kappa_score = roc_auc_score = accuracy_score = None
try:
    from lifelines.utils import concordance_index
except ImportError:                                                 # pragma: no cover
    concordance_index = None


# ---------- Inference dataset ----------

class CohortInferenceDataset(torch.utils.data.Dataset):
    """Read pretrain_corpus.parquet rows for one cohort and yield (gene_ids, bin_ids) tensors."""

    def __init__(
        self,
        corpus_parquet: Path,
        metadata_parquet: Path,
        tokenizer_json: Path,
        cohort: str,
        rna_seq_len: int = 4200,
    ) -> None:
        meta = pd.read_parquet(metadata_parquet)
        cohort_sids = meta.loc[meta["cohort"] == cohort, "sample_id"].astype(str).tolist()
        df = pd.read_parquet(corpus_parquet)
        df = df[df["sample_id"].astype(str).isin(cohort_sids)]
        with open(tokenizer_json) as f:
            tok = json.load(f)
        bin_offset = int(tok.get("bin_offset", 2))
        gene_to_id = ({k: int(v) for k, v in tok["gene_to_id"].items()}
                      if "gene_to_id" in tok
                      else {g: i + bin_offset + 1 for i, g in enumerate(tok.get("vocab", []))})
        sample_to_g: dict[str, np.ndarray] = {}
        sample_to_b: dict[str, np.ndarray] = {}
        for sid, sub in df.groupby("sample_id"):
            gids = np.array([gene_to_id.get(g, 0) for g in sub["gene"]], dtype=np.int64)
            bids = np.array(sub["rank_bin"], dtype=np.int64) + bin_offset
            sample_to_g[str(sid)] = gids
            sample_to_b[str(sid)] = bids
        self.sample_ids = sorted(sample_to_g.keys())
        self.gene_ids = sample_to_g
        self.bin_ids = sample_to_b
        self.rna_seq_len = rna_seq_len
        self.cohort = cohort
        print(f"[evaluate] {cohort}: {len(self.sample_ids)} samples loaded")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sid = self.sample_ids[idx]
        L = self.rna_seq_len
        out_g = np.zeros(L, dtype=np.int64)
        out_b = np.zeros(L, dtype=np.int64)
        out_mask = np.ones(L, dtype=bool)                            # True = pad
        out_g[0] = 2                                                # CLS
        n = min(L - 1, len(self.gene_ids[sid]))
        out_g[1:1 + n] = self.gene_ids[sid][:n]
        out_b[1:1 + n] = self.bin_ids[sid][:n]
        out_mask[:1 + n] = False
        return {
            "sample_id": sid,
            "gene_ids": torch.from_numpy(out_g),
            "bin_ids": torch.from_numpy(out_b),
            "attention_mask": torch.from_numpy(out_mask),
        }


# ---------- Inference + prediction extraction ----------

def _load_finetuned_model(ckpt_path: Path, device: torch.device):
    """Manually instantiate MultiModalHemeFM and load weights from ckpt (skipping LightningModule wrapper)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hemefm.models.multimodal import MultiModalHemeFM, MultiModalConfig
    from hemefm.models.transformer import HemeFMEncoder

    cfg_mm = MultiModalConfig(
        n_subtypes=6, n_eln_classes=3, n_drugs=50,
        n_mutation_genes=200, n_variant_classes=6,
        n_methylation_features=5000,
        fusion_layers=2, fusion_heads=12, dropout=0.1, freeze_rna_encoder=False,
    )
    rna = HemeFMEncoder(
        vocab_size=4150, n_bins=16, max_seq_len=4200,
        d_model=768, n_heads=12, n_layers=24, d_ff=3072,
        dropout=0.1, attention_dropout=0.1,
        mask_token_id=1, pad_token_id=0, cls_token_id=2,
    )
    model = MultiModalHemeFM(rna_encoder=rna, cfg=cfg_mm).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    clean_sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"[evaluate] loaded {len(clean_sd) - len(missing)}/{len(clean_sd)} tensors; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")

    # Return a thin shim so downstream `lm.model` still works
    class _Shim:
        def __init__(self, m): self.model = m
        def eval(self): self.model.eval(); return self
    return _Shim(model)


@torch.no_grad()
def run_inference(
    ckpt_path: Path,
    corpus_parquet: Path,
    metadata_parquet: Path,
    tokenizer_json: Path,
    cohort: str,
    output_dir: Path,
    rna_seq_len: int = 4200,
    batch_size: int = 8,
    n_mutation_genes: int = 200,
    max_mutations: int = 12,
    n_methylation_features: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")

    lm = _load_finetuned_model(ckpt_path, device)
    model = lm.model
    model.eval()

    ds = CohortInferenceDataset(corpus_parquet, metadata_parquet, tokenizer_json, cohort, rna_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)

    all_preds: list[dict] = []
    all_embeds: list[np.ndarray] = []
    all_sids: list[str] = []

    for batch in loader:
        sids = batch.pop("sample_id")
        # Add dummy modalities to match multimodal expected signature
        bs = batch["gene_ids"].shape[0]
        batch["mut_gene_ids"] = torch.zeros((bs, max_mutations), dtype=torch.long, device=device)
        batch["mut_variant_class_ids"] = torch.zeros((bs, max_mutations), dtype=torch.long, device=device)
        batch["mut_attention_mask"] = torch.ones((bs, max_mutations), dtype=torch.bool, device=device)
        batch["methylation"] = torch.full((bs, n_methylation_features), float("nan"), device=device)
        for k in ("gene_ids", "bin_ids", "attention_mask"):
            batch[k] = batch[k].to(device)

        out = model(batch)
        all_sids.extend(sids if isinstance(sids, list) else sids.tolist())
        # Actual MultiModalHemeFM.forward returns:
        #   {"subtype": {"logits"/"probs"}, "eln": {"logit"/"probs"}, "survival": {"log_risk"}, "drug": {"pred"}}
        for i in range(bs):
            row = {"sample_id": all_sids[-bs + i]}
            if "subtype" in out and isinstance(out["subtype"], dict):
                sb = out["subtype"]
                if "probs" in sb:
                    probs = sb["probs"][i].float().cpu().numpy()
                    row["subtype_pred"] = int(np.argmax(probs))
                    row["subtype_probs"] = json.dumps(probs.tolist())
                elif "logits" in sb:
                    logits = sb["logits"][i].float().cpu().numpy()
                    row["subtype_pred"] = int(np.argmax(logits))
                    row["subtype_probs"] = json.dumps(logits.tolist())
            if "eln" in out and isinstance(out["eln"], dict):
                el = out["eln"]
                if "probs" in el:
                    probs = el["probs"][i].float().cpu().numpy()
                    row["eln_pred"] = int(np.argmax(probs))
                    row["eln_probs"] = json.dumps(probs.tolist())
                elif "logit" in el:
                    logit = float(el["logit"][i].cpu().numpy())
                    # Cumulative-link single logit → quantile-binned class
                    row["eln_pred"] = 0 if logit < -0.5 else (1 if logit < 0.5 else 2)
                    row["eln_logit"] = logit
            if "survival" in out and isinstance(out["survival"], dict):
                lr = out["survival"].get("log_risk")
                if lr is not None:
                    row["os_risk"] = float(lr[i].cpu().numpy())
            if "drug" in out and isinstance(out["drug"], dict):
                pred = out["drug"].get("pred")
                if pred is not None:
                    row["drug_response"] = json.dumps(pred[i].float().cpu().numpy().tolist())
            all_preds.append(row)

    preds_df = pd.DataFrame(all_preds)
    preds_df.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)
    print(f"[evaluate] wrote {output_dir/'predictions.tsv'} ({len(preds_df)} rows)")

    embeds_df = pd.DataFrame()
    if all_embeds:
        emb = np.concatenate(all_embeds, axis=0)
        embeds_df = pd.DataFrame(emb, index=all_sids)
        embeds_df.index.name = "sample_id"
        embeds_df.to_parquet(output_dir / "embeddings.parquet")
        print(f"[evaluate] wrote {output_dir/'embeddings.parquet'} ({emb.shape})")

    return preds_df, embeds_df


# ---------- Metric computation against external labels ----------

def compute_metrics(preds: pd.DataFrame, labels_tsv: Optional[Path]) -> dict:
    if labels_tsv is None or not labels_tsv.exists():
        return {"note": "no labels provided, skipping metric computation"}
    labels = pd.read_csv(labels_tsv, sep="\t")
    if "sample_id" not in labels.columns:
        raise ValueError("labels TSV must have a 'sample_id' column")
    merged = preds.merge(labels, on="sample_id", how="inner")
    print(f"[metrics] {len(merged)} samples have both prediction + label")

    results: dict = {"n_matched": int(len(merged))}

    # Subtype: accuracy + per-class
    if "subtype_label" in merged.columns and accuracy_score is not None:
        mask = merged["subtype_label"].notna()
        if mask.sum() > 0:
            results["subtype_acc"] = float(accuracy_score(
                merged.loc[mask, "subtype_label"].astype(int),
                merged.loc[mask, "subtype_pred"].astype(int),
            ))

    # ELN: QWK
    if "eln_label" in merged.columns and cohen_kappa_score is not None:
        mask = merged["eln_label"].notna()
        if mask.sum() > 1:
            results["eln_qwk"] = float(cohen_kappa_score(
                merged.loc[mask, "eln_label"].astype(int),
                merged.loc[mask, "eln_pred"].astype(int),
                weights="quadratic",
            ))

    # Survival: Harrell's C-index
    if all(c in merged.columns for c in ["os_time", "os_event"]) and concordance_index is not None:
        mask = merged["os_time"].notna() & merged["os_event"].notna() & merged["os_risk"].notna()
        if mask.sum() > 1:
            results["os_c_index"] = float(concordance_index(
                merged.loc[mask, "os_time"].astype(float),
                -merged.loc[mask, "os_risk"].astype(float),               # higher risk → shorter survival
                merged.loc[mask, "os_event"].astype(int),
            ))

    # Drug: Spearman ρ averaged across drugs (if labels are present as drug_<name> columns)
    drug_cols = [c for c in merged.columns if c.startswith("drug_") and c not in ("drug_response_pred",)]
    if drug_cols:
        spearman_rhos = []
        # drug_response is JSON list; parse and align
        pred_drug_lists = merged["drug_response"].apply(json.loads).tolist()
        n_drugs = len(pred_drug_lists[0])
        from scipy.stats import spearmanr
        for d_idx in range(n_drugs):
            preds_d = np.array([r[d_idx] for r in pred_drug_lists])
            if d_idx < len(drug_cols):
                labels_d = merged[drug_cols[d_idx]].astype(float)
                m = labels_d.notna() & np.isfinite(preds_d)
                if m.sum() > 5:
                    rho, _ = spearmanr(preds_d[m], labels_d[m])
                    spearman_rhos.append(rho)
        if spearman_rhos:
            results["drug_spearman_mean"] = float(np.nanmean(spearman_rhos))
            results["drug_spearman_n_drugs"] = int(len(spearman_rhos))

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--corpus-parquet", type=Path,
                   default=Path("data/processed/pretrain_corpus.parquet"))
    p.add_argument("--metadata-parquet", type=Path,
                   default=Path("data/processed/pretrain_corpus_metadata.parquet"))
    p.add_argument("--tokenizer-json", type=Path,
                   default=Path("data/processed/tokenizer.json"))
    p.add_argument("--cohort", required=True,
                   help="Cohort tag as in metadata.parquet (TCGA-LAML, TARGET-AML, GSE6891, GSE37642, BeatAML2, GTEx-WB).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--labels-tsv", type=Path, default=None,
                   help="Optional sample_id-keyed TSV with eln_label / subtype_label / os_time / os_event / drug_<name> columns.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rna-seq-len", type=int, default=4200)
    args = p.parse_args()

    preds, _embeds = run_inference(
        args.ckpt, args.corpus_parquet, args.metadata_parquet, args.tokenizer_json,
        args.cohort, args.output_dir,
        rna_seq_len=args.rna_seq_len, batch_size=args.batch_size,
    )
    metrics = compute_metrics(preds, args.labels_tsv)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[metrics]", json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
