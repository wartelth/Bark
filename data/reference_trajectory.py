"""
Build reference trajectories from jacket CSV for reward shaping or IL.
Output: .npy files with (T, dim) or (T, n_legs, dim) suitable for comparison in the env.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from data.jacket_loader import load_jacket_csv


def jacket_to_reference(
    csv_path: str | Path,
    out_path: Optional[str | Path] = None,
    sep: str = ";",
    normalize: bool = True,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """
    Load jacket CSV and save a reference trajectory array.
    Returns array of shape (T, n_features + n_targets) or (T, 2, 9) for [features, target] per step.
    If out_path is set, saves as .npy and returns the array.
    """
    X, y = load_jacket_csv(csv_path, sep=sep)
    if max_steps is not None:
        X, y = X[:max_steps], y[:max_steps]

    if normalize:
        # Scale to roughly [-1, 1] per channel for sim compatibility
        all_ = np.concatenate([X, y], axis=1)
        lo, hi = np.percentile(all_, [1, 99], axis=0)
        span = np.where(hi - lo > 1e-8, hi - lo, 1.0)
        X = (X - lo[:X.shape[1]]) / span[:X.shape[1]]
        y = (y - lo[X.shape[1]:]) / span[X.shape[1]:]

    # Single trajectory: each row is [X, y] flattened, or we store (T, 2, 9)
    ref = np.concatenate([X, y], axis=1).astype(np.float32)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, ref)
    return ref
