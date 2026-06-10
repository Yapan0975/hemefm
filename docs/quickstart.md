# Quickstart troubleshooting

## 5090-specific notes

The RTX 5090 uses Blackwell (sm_120). Three install paths in increasing severity:

1. **Stable cu126 wheels (try first):**
   ```
   uv pip install torch lightning hydra-core wandb
   ```
   Works for RTX 30 / 40 / many 5090 setups on driver ≥ 555.

2. **Nightly cu128 (fall back if sm_120 errors):**
   ```
   uv pip install --pre --index https://download.pytorch.org/whl/nightly/cu128 \
       torch torchvision torchaudio
   ```

3. **Compile-from-source (last resort):** only if both wheels above fail.

To confirm the install:
```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # should print (12, 0) for 5090
```

## Flash Attention 2

- **Linux**: `uv pip install flash-attn --no-build-isolation`. Needs the matching CUDA toolkit.
- **Windows**: PyTorch's `scaled_dot_product_attention` auto-selects Flash on Blackwell. Do not install `flash-attn` directly.

## W&B

```
wandb login                    # interactive
export WANDB_API_KEY=...        # CI / non-interactive
export WANDB_MODE=offline       # cluster without internet — sync later with `wandb sync`
```

## Hydra CLI cheat sheet

```
# Override single keys
uv run hemefm-train trainer.max_epochs=5 data.batch_size=32

# Switch composition groups
uv run hemefm-train model=hemefm_base data=beataml

# Multirun sweep
uv run hemefm-train -m model.d_model=64,128,256

# See the resolved config without training
uv run hemefm-train --cfg job --resolve
```

## Common errors

| Symptom | Fix |
|---|---|
| `RuntimeError: CUDA error: no kernel image is available` | sm_120 mismatch — install nightly cu128 wheel |
| `OSError: [WinError 1455] The paging file is too small` | Windows pagefile — set Trainer `num_workers=0` or extend virtual memory |
| `W&B API key is missing` | `wandb login` or `export WANDB_MODE=offline` |
| `Hydra config not found` | run from the repo root: `cd D:\_7_sci\hu\manu01\hemefm` |
| `ImportError: flash_attn` | not needed — install the `[flash]` extra only if you specifically want it |
