"""
Train a supervised prosthetic prediction network:
    f(obs_3leg) -> action_leg3

Input: 3-leg observation from teacher rollouts.
Target: teacher's leg-3 action (joint position targets for hip, thigh, calf).
Loss: MSE.

Usage:
    PYTHONPATH=. python train/train_supervised.py
    PYTHONPATH=. python train/train_supervised.py --config configs/supervised_go1.yaml
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "teacher_rollouts.npz"
SAVE_DIR = REPO / "models" / "supervised_prosthetic"


class ProstheticMLP(nn.Module):
    """Predicts leg-3 action (3D) from 3-leg observation."""

    def __init__(self, obs_dim: int, action_dim: int = 3, hidden: list[int] = [256, 256, 128]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(path: Path, val_fraction: float = 0.1):
    data = np.load(path)
    X = torch.from_numpy(data["obs_3leg"])
    y = torch.from_numpy(data["action_leg3"])
    print(f"Loaded {len(X)} samples: X={X.shape}, y={y.shape}")

    dataset = TensorDataset(X, y)
    return split_dataset(dataset, val_fraction=val_fraction)


def split_dataset(dataset: Dataset, val_fraction: float = 0.1):
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val])


def make_loaders(train_ds, val_ds, batch_size: int, num_workers: int, pin_memory: bool):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def fit_model(
    train_ds,
    val_ds,
    save_dir: Path,
    hidden: list[int] = [256, 256, 128],
    lr: float = 3e-4,
    batch_size: int = 1024,
    epochs: int = 30,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    patience: int = 8,
    initial_state_dict: dict | None = None,
    onnx_name: str = "prosthetic.onnx",
):
    save_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    pin_memory = use_cuda
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size, num_workers, pin_memory)

    obs_dim = train_ds[0][0].shape[0]
    action_dim = train_ds[0][1].shape[0]
    print(f"obs_dim={obs_dim}, action_dim={action_dim}, device={device}")

    raw_model = ProstheticMLP(obs_dim, action_dim, hidden).to(device)
    if initial_state_dict is not None:
        raw_model.load_state_dict(initial_state_dict)
    model = raw_model
    has_triton = importlib.util.find_spec("triton") is not None
    if use_cuda and has_triton and hasattr(torch, "compile"):
        try:
            model = torch.compile(raw_model)
        except Exception:
            model = raw_model
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=pin_memory)
                y_batch = y_batch.to(device, non_blocking=pin_memory)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item() * len(X_batch)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(raw_model.state_dict(), save_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs (patience={patience}).")
                break

    torch.save(raw_model.state_dict(), save_dir / "final_model.pt")

    raw_model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    raw_model.eval()
    dummy = torch.randn(1, obs_dim).to(device)
    try:
        torch.onnx.export(
            raw_model, dummy, str(save_dir / onnx_name),
            input_names=["obs_3leg"], output_names=["action_leg3"],
            dynamic_axes={"obs_3leg": {0: "batch"}, "action_leg3": {0: "batch"}},
        )
    except ModuleNotFoundError as e:
        print(f"Skipping ONNX export ({e}). PyTorch checkpoint was saved successfully.")
    training_log = {
        "hidden": list(hidden),
        "lr": lr,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_val_loss": float(best_val_loss),
        "device": device,
        "num_workers": num_workers,
        "patience": patience,
        "history": history,
    }
    (save_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))
    print(f"\nBest val_loss: {best_val_loss:.6f}")
    print(f"Saved to {save_dir}")
    return raw_model, training_log


def train(
    data_path: Path = DEFAULT_DATA,
    hidden: list[int] = [256, 256, 128],
    lr: float = 3e-4,
    batch_size: int = 1024,
    epochs: int = 30,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    patience: int = 8,
    save_dir: Path = SAVE_DIR,
):
    train_ds, val_ds = load_data(data_path)
    return fit_model(
        train_ds=train_ds,
        val_ds=val_ds,
        save_dir=save_dir,
        hidden=hidden,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        num_workers=num_workers,
        patience=patience,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    kwargs = dict(
        hidden=[256, 256, 128],
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        patience=args.patience,
    )
    if args.data:
        kwargs["data_path"] = Path(args.data)
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        kwargs.update({k: v for k, v in cfg.items() if k in kwargs})
        if "data_path" in cfg and not args.data:
            kwargs["data_path"] = Path(cfg["data_path"])

    train(**kwargs)


if __name__ == "__main__":
    main()
