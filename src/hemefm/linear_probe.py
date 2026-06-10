"""§4.3 Resolution arm C: RNA-only linear probe on frozen pretrained encoder.

Tests whether the pretraining-derived foundation-model CLS embedding transfers
across the BeatAML (training cohort) → GSE6891 (held-out microarray cohort)
platform shift more cleanly than the full multimodal cross-attention model.

Pipeline:
  1. Load frozen pretrained hemefm_base RNA encoder (epoch=29 ckpt from §4.1).
  2. Extract 768-dim CLS embedding for every sample in BeatAML2 train+val and
     GSE6891 (using the tokenizer-aligned rank-bin parquet from §4.3 prep).
  3. Train a sklearn LogisticRegression on BeatAML train embeddings → ELN label.
  4. Predict on BeatAML val (within-cohort sanity check) and GSE6891 (true external).
  5. Report ELN QWK, accuracy, per-class AUROC, Spearman of expected score.

Compared with the §4.2 PCA-64 baseline:
  - PCA-64: linear projection of rank-binned counts → 3-class logreg
  - this:   pretrained transformer encoder pooled to CLS → 3-class logreg

If linear-probe ELN QWK on GSE6891 > the full-finetune QWK (0.000 argmax,
Spearman 0.21), the §3.13 Stage 2'' resolution arm C succeeds and the
foundation-model embedding IS more platform-robust than the full-task model.

Usage:
    python -m hemefm.linear_probe \\
        --pretrain-ckpt logs/pretrain-v3-base-ddp-2661samples/version_0/checkpoints/epoch=29-step=2280.ckpt \\
        --train-cohort BeatAML2 \\
        --test-cohorts GSE6891 \\
        --output-dir outputs/linear_probe_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _load_pretrained_rna(ckpt_path: Path, device: torch.device):
    """Load just the RNA encoder weights from a pretraining checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hemefm.models.transformer import HemeFMEncoder

    rna = HemeFMEncoder(
        vocab_size=4150, n_bins=16, max_seq_len=4200,
        d_model=768, n_heads=12, n_layers=24, d_ff=3072,
        dropout=0.1, attention_dropout=0.1,
        mask_token_id=1, pad_token_id=0, cls_token_id=2,
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # MFM ckpt structure: keys are like "encoder.<...>" (no model. prefix)
    encoder_sd: dict = {}
    for k, v in sd.items():
        if k.startswith("encoder.") and not k.startswith("encoder.encoder.layers.0.norm"):
            # encoder.<...> in the MFM module corresponds to rna_encoder.<...> after strip "encoder."
            encoder_sd[k[len("encoder."):]] = v
        elif k.startswith("model.encoder."):
            encoder_sd[k[len("model.encoder."):]] = v
    if not encoder_sd:
        # Try: keys are directly hemefm encoder field names
        encoder_sd = {k: v for k, v in sd.items() if not k.startswith("mfm_head.")}
    missing, unexpected = rna.load_state_dict(encoder_sd, strict=False)
    print(f"[linear_probe] RNA encoder load: {len(encoder_sd) - len(missing)}/{len(encoder_sd)} tensors; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    return rna


@torch.no_grad()
def extract_cls_embeddings(rna_encoder, dataset, batch_size: int = 4, device=None) -> tuple[np.ndarray, list[str]]:
    """For each sample in dataset, run RNA encoder transformer and return CLS hidden state."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    all_emb: list[np.ndarray] = []
    all_sids: list[str] = []
    for batch in loader:
        sids = batch.pop("sample_id")
        g = batch["gene_ids"].to(device)
        b = batch["bin_ids"].to(device)
        am = batch["attention_mask"].to(device)
        # Use just the transformer encoder layers (.encoder access mirrors multimodal._encode_rna)
        x = rna_encoder.gene_emb(g) + rna_encoder.bin_emb(b)
        x = rna_encoder.embed_norm(x)
        x = rna_encoder.embed_dropout(x)
        h = rna_encoder.encoder(x, src_key_padding_mask=am)
        cls = h[:, 0].float().cpu().numpy()                  # (bs, 768)
        all_emb.append(cls)
        all_sids.extend(sids if isinstance(sids, list) else sids.tolist())
    return np.concatenate(all_emb, axis=0), all_sids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-ckpt", type=Path, required=True)
    p.add_argument("--train-corpus-parquet", type=Path,
                   default=Path("data/processed/pretrain_corpus.parquet"))
    p.add_argument("--train-metadata-parquet", type=Path,
                   default=Path("data/processed/pretrain_corpus_metadata.parquet"))
    p.add_argument("--train-clinical-xlsx", type=Path,
                   default=Path("data/raw/beataml_v2_biodev/beataml_wv1to4_clinical.xlsx"))
    p.add_argument("--tokenizer-json", type=Path,
                   default=Path("data/processed/tokenizer.json"))
    p.add_argument("--external-dirs", nargs="+", default=[],
                   help="Each ext dir must contain expression.parquet + metadata.parquet + labels.tsv")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hemefm.evaluate import CohortInferenceDataset
    from hemefm.data.beataml_multimodal import _derive_labels
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score
    from scipy.stats import spearmanr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[linear_probe] device: {device}")

    # 1. Load frozen RNA encoder
    rna = _load_pretrained_rna(args.pretrain_ckpt, device)

    # 2. Extract embeddings for BeatAML2 (train+val from finetune split)
    print("\n[linear_probe] extracting BeatAML2 embeddings...")
    ba_ds = CohortInferenceDataset(args.train_corpus_parquet, args.train_metadata_parquet,
                                    args.tokenizer_json, "BeatAML2", rna_seq_len=4200)
    ba_emb, ba_sids = extract_cls_embeddings(rna, ba_ds, args.batch_size, device)
    print(f"  BeatAML2 embeddings: {ba_emb.shape}")

    # Match to ELN labels
    clinical = pd.read_excel(args.train_clinical_xlsx)
    labels = _derive_labels(clinical)
    ba_eln = pd.Series([labels.loc[s, "eln_label"] if s in labels.index else np.nan
                         for s in ba_sids], index=ba_sids).astype(float)
    mask = ba_eln.notna()
    print(f"  BeatAML2 ELN-labeled: {mask.sum()} of {len(ba_sids)}")

    X = ba_emb[mask.values]
    y = ba_eln[mask].astype(int).values

    # 80/20 split (same seed as Phase 4')
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(0.20 * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"  train n={len(train_idx)}, val n={len(val_idx)}")

    # 3. Train logistic regression
    print("\n[linear_probe] fitting LogisticRegression on BeatAML train...")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[train_idx], y[train_idx])

    results: dict = {
        "model": "frozen hemefm_base 173M RNA encoder CLS + sklearn LogisticRegression",
        "BeatAML2_val": {},
        "external_cohorts": {},
    }

    # 4. Predict on BeatAML val
    y_val_pred = clf.predict(X[val_idx])
    y_val_proba = clf.predict_proba(X[val_idx])
    results["BeatAML2_val"]["n"] = int(len(val_idx))
    results["BeatAML2_val"]["eln_qwk"] = float(cohen_kappa_score(y[val_idx], y_val_pred, weights="quadratic"))
    results["BeatAML2_val"]["eln_acc"] = float(accuracy_score(y[val_idx], y_val_pred))
    results["BeatAML2_val"]["pred_dist"] = np.bincount(y_val_pred, minlength=3).tolist()
    results["BeatAML2_val"]["true_dist"] = np.bincount(y[val_idx], minlength=3).tolist()
    expected = (y_val_proba * np.array([0, 1, 2])).sum(1)
    results["BeatAML2_val"]["spearman_expected_vs_true"] = float(spearmanr(expected, y[val_idx]).statistic)
    print(f"  BeatAML2 val: ELN QWK = {results['BeatAML2_val']['eln_qwk']:.4f}, acc = {results['BeatAML2_val']['eln_acc']:.4f}")

    # 5. Predict on external cohorts
    for ext_dir_str in args.external_dirs:
        ext_dir = Path(ext_dir_str)
        cohort = ext_dir.name
        print(f"\n[linear_probe] external cohort: {cohort}")
        ext_ds = CohortInferenceDataset(ext_dir / "expression.parquet", ext_dir / "metadata.parquet",
                                          args.tokenizer_json, cohort, rna_seq_len=4200)
        ext_emb, ext_sids = extract_cls_embeddings(rna, ext_ds, args.batch_size, device)
        print(f"  {cohort} embeddings: {ext_emb.shape}")
        ext_lab = pd.read_csv(ext_dir / "labels.tsv", sep="\t")
        ext_lab = ext_lab.set_index("sample_id")
        ext_eln = pd.Series([ext_lab.loc[s, "eln_label"] if s in ext_lab.index else np.nan
                              for s in ext_sids], index=ext_sids).astype(float)
        m_ext = ext_eln.notna()
        if m_ext.sum() < 5:
            print(f"  too few labeled ({m_ext.sum()}); skipping")
            continue
        X_ext = ext_emb[m_ext.values]
        y_ext = ext_eln[m_ext].astype(int).values
        y_ext_pred = clf.predict(X_ext)
        y_ext_proba = clf.predict_proba(X_ext)
        ext_res = {
            "n": int(m_ext.sum()),
            "eln_qwk": float(cohen_kappa_score(y_ext, y_ext_pred, weights="quadratic")),
            "eln_acc": float(accuracy_score(y_ext, y_ext_pred)),
            "pred_dist": np.bincount(y_ext_pred, minlength=3).tolist(),
            "true_dist": np.bincount(y_ext, minlength=3).tolist(),
            "spearman_expected_vs_true": float(spearmanr((y_ext_proba * np.array([0,1,2])).sum(1), y_ext).statistic),
        }
        # Per-class AUROC
        for c in [0, 1, 2]:
            y_bin = (y_ext == c).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                ext_res[f"auroc_class{c}"] = float(roc_auc_score(y_bin, y_ext_proba[:, c]))
        results["external_cohorts"][cohort] = ext_res
        print(f"  {cohort} ELN QWK = {ext_res['eln_qwk']:.4f}, acc = {ext_res['eln_acc']:.4f}, Spearman = {ext_res['spearman_expected_vs_true']:.4f}")

    (args.output_dir / "linear_probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[linear_probe] all results written to {args.output_dir/'linear_probe_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
