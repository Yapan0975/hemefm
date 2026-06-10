"""HemeFM training entrypoint — Hydra + PyTorch Lightning.

Two modes selected via the `mode:` config key (default: pretrain).

Pretrain (MFM, single-task):
    uv run hemefm-train experiment=hello_world
    uv run python -m hemefm.train experiment=hello_world model.d_model=128

Multi-modal multi-task fine-tune:
    uv run hemefm-train experiment=finetune_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = (Path(__file__).resolve().parent.parent.parent / "configs").as_posix()


def _max_steps(cfg: DictConfig, default_n_per_epoch: int = 1) -> int:
    max_epochs = int(cfg.trainer.max_epochs)
    n_train = int(cfg.data.get("n_train", default_n_per_epoch))
    batch_size = int(cfg.data.get("batch_size", 1))
    return max_epochs * max(1, n_train // batch_size)


def _run_pretrain(cfg: DictConfig) -> int:
    from hemefm.lightning_modules import MFMLightningModule

    encoder = hydra.utils.instantiate(cfg.model)
    print(f"[info] pretrain model: {type(encoder).__name__} with {encoder.num_parameters():,} parameters")

    datamodule = hydra.utils.instantiate(cfg.data)

    lightning_module = MFMLightningModule(
        encoder=encoder,
        lr=3e-4,
        weight_decay=0.05,
        warmup_steps=20,
        max_steps=_max_steps(cfg),
    )

    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.fit(lightning_module, datamodule=datamodule)
    return 0



def _run_finetune_adversarial(cfg):
    import hydra
    from hemefm.lightning_modules.finetune_adversarial import FinetuneAdversarialLightningModule
    model = hydra.utils.instantiate(cfg.model, _recursive_=True)
    pretrained = cfg.get("pretrained_ckpt", None)
    if pretrained:
        import torch
        from pathlib import Path
        ckpt_path = Path(str(pretrained))
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            backbone_sd = {}
            for k, v in sd.items():
                clean_k = k.replace("encoder.", "", 1) if k.startswith("encoder.") else k
                clean_k = clean_k.replace("model.encoder.", "", 1) if clean_k.startswith("model.encoder.") else clean_k
                backbone_sd["rna_encoder." + clean_k] = v
            missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
            print(f"[info] loaded {len(backbone_sd) - len(missing)}/{len(backbone_sd)} pretrained tensors")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] adversarial finetune model: {type(model).__name__} {n_total:,} params")
    datamodule = hydra.utils.instantiate(cfg.data)
    lm = FinetuneAdversarialLightningModule(
        model=model, adv_lambda=cfg.get("adv_lambda", 0.1), adv_hidden=cfg.get("adv_hidden", 256),
        weighting=cfg.get("weighting", "kendall"), lr=cfg.get("lr", 1e-4),
        encoder_lr=cfg.get("encoder_lr", 1e-5), weight_decay=cfg.get("weight_decay", 0.05),
        warmup_steps=cfg.get("warmup_steps", 50), max_steps=_max_steps(cfg),
        unfreeze_after_step=cfg.get("unfreeze_after_step", None),
    )
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.fit(lm, datamodule=datamodule)
    return 0


def _run_finetune(cfg: DictConfig) -> int:
    from hemefm.lightning_modules import FinetuneLightningModule

    model = hydra.utils.instantiate(cfg.model, _recursive_=True)

    # Load pretrained backbone if specified
    pretrained = cfg.get("pretrained_ckpt", None)
    if pretrained:
        import torch
        from pathlib import Path
        ckpt_path = Path(str(pretrained))
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            # Strip Lightning prefix and remap mfm-model keys to rna_encoder.*
            backbone_sd = {}
            for k, v in sd.items():
                # MFM ckpt structure: encoder.<...> or model.encoder.<...>
                clean_k = k.replace("encoder.", "", 1) if k.startswith("encoder.") else k
                clean_k = clean_k.replace("model.encoder.", "", 1) if clean_k.startswith("model.encoder.") else clean_k
                # Target into rna_encoder of multimodal model
                backbone_sd["rna_encoder." + clean_k] = v
            missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
            n_loaded = sum(1 for k in backbone_sd if not any(k == m for m in missing))
            print(f"[info] loaded {n_loaded}/{len(backbone_sd)} pretrained tensors from {ckpt_path.name}")
            print(f"[info]   missing in model (still random init): {len(missing)}")
            print(f"[info]   unexpected in ckpt (skipped): {len(unexpected)}")
        else:
            print(f"[warn] pretrained_ckpt path does not exist: {ckpt_path}")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] finetune model: {type(model).__name__} with {n_trainable:,}/{n_total:,} trainable params")

    datamodule = hydra.utils.instantiate(cfg.data)

    lightning_module = FinetuneLightningModule(
        model=model,
        weighting=cfg.get("weighting", "kendall"),
        lr=cfg.get("lr", 1e-4),
        encoder_lr=cfg.get("encoder_lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0.05),
        warmup_steps=cfg.get("warmup_steps", 20),
        max_steps=_max_steps(cfg),
        unfreeze_after_step=cfg.get("unfreeze_after_step", None),
    )

    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.fit(lightning_module, datamodule=datamodule)
    return 0


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> int:
    L.seed_everything(cfg.seed, workers=True)

    print("=" * 70)
    print(f"HemeFM run  (mode={cfg.get('mode', 'pretrain')}, project={cfg.project_name}, seed={cfg.seed})")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 70)

    mode = cfg.get("mode", "pretrain")
    try:
        if mode == "pretrain":
            rc = _run_pretrain(cfg)
        elif mode == "finetune_adversarial":
            rc = _run_finetune_adversarial(cfg)
        elif mode == "finetune":
            rc = _run_finetune(cfg)
        else:
            raise ValueError(f"unknown mode: {mode}")
    except Exception as exc:                  # noqa: BLE001
        print(f"[smoke-test FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1

    print()
    print(f"[smoke-test PASS] {mode} completed without errors.")
    print(f"[smoke-test PASS] outputs at: {Path.cwd() / 'outputs'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
