"""P0 reviewer-fix experiments: discriminator AUC recompute + external PCA baseline + paired bootstrap."""
import sys
import math
import json
from pathlib import Path
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch

# ============================================================
# P0-5: External PCA baseline on GSE6891
# ============================================================
print('=' * 60)
print('P0-5: External PCA baseline on GSE6891')
print('=' * 60)

# Load BeatAML training data (n=347) and labels
# Use saved train/val splits if present
beataml = pd.read_parquet('data/finetune/beataml_multimodal_v3.parquet') if Path('data/finetune/beataml_multimodal_v3.parquet').exists() else None
if beataml is None:
    # Fallback: load from raw
    print('Loading BeatAML from rank-binned corpus...')
    import sys
    sys.path.insert(0, 'src')
    from hemefm.data.beataml import BeatAMLDataModule

import sklearn.decomposition, sklearn.linear_model
from sklearn.metrics import cohen_kappa_score

# Load BeatAML expression + ELN labels + train/val/test indices
# Look for the saved finetune data
candidates = list(Path('data/finetune').glob('*beataml*.parquet'))
print(f'  Candidate finetune parquets: {[str(c) for c in candidates]}')
