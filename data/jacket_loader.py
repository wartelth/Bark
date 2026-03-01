"""
Load jacket (IMU) CSV data: IMU1–IMU3 as features, IMU4 as target.
Format: columns like IMU1AccelX, IMU1AccelY, ... IMU4AccelX, ... (semicolon or comma separated).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "IMU1AccelX", "IMU1AccelY", "IMU1AccelZ",
    "IMU1GyroX", "IMU1GyroY", "IMU1GyroZ",
    "IMU2AccelX", "IMU2AccelY", "IMU2AccelZ",
    "IMU2GyroX", "IMU2GyroY", "IMU2GyroZ",
    "IMU3AccelX", "IMU3AccelY", "IMU3AccelZ",
    "IMU3GyroX", "IMU3GyroY", "IMU3GyroZ",
]
TARGET_COLS = [
    "IMU4AccelX", "IMU4AccelY", "IMU4AccelZ",
    "IMU4GyroX", "IMU4GyroY", "IMU4GyroZ",
]


def load_jacket_csv(
    path: str | Path,
    sep: str = ";",
    feature_cols: Optional[list[str]] = None,
    target_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load jacket CSV and return (X, y) as numpy arrays.
    X: features from IMU1–IMU3, y: target from IMU4.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=sep)
    # Normalize column names (strip spaces)
    df.columns = df.columns.str.strip()
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS

    missing_f = [c for c in feature_cols if c not in df.columns]
    missing_t = [c for c in target_cols if c not in df.columns]
    if missing_f or missing_t:
        raise ValueError(
            f"Missing columns: features {missing_f}, targets {missing_t}. "
            f"Available: {list(df.columns)}"
        )

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    return X, y


def jacket_to_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 32,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (X, y) to sliding-window sequences for LSTM/IL.
    Returns (X_seq, y_seq) with shapes (n_seq, seq_len, n_feat), (n_seq, seq_len, n_target).
    """
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0, seq_len, y.shape[1]))

    n_seq = (n - seq_len) // stride + 1
    X_seq = np.zeros((n_seq, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n_seq, seq_len, y.shape[1]), dtype=np.float32)
    for i in range(n_seq):
        start = i * stride
        end = start + seq_len
        X_seq[i] = X[start:end]
        y_seq[i] = y[start:end]
    return X_seq, y_seq
