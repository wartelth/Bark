"""
Train an explicit imitation-learning student with DAgger-style aggregation.

The supervised trainer remains the pure behavior-cloning baseline.
This script starts from teacher data, then repeatedly:
  1. rolls the current student in the hybrid env,
  2. queries the teacher for the correct leg-3 action,
  3. aggregates those labels,
  4. re-trains the student on the larger dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.scenario_library import apply_scenario, sample_scenario, scenario_pool
from pretrained.load_teacher import GO1_LEG3_JOINT_IX, load_teacher, make_go1_env, split_obs_and_action
from train.train_supervised import DEFAULT_DATA, ProstheticMLP, fit_model, split_dataset

REPO = Path(__file__).resolve().parent.parent
SAVE_DIR = REPO / "models" / "imitation_prosthetic"


def load_npz_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    result = {key: data[key] for key in data.files}
    print(f"Loaded base dataset from {path}: {result['obs_3leg'].shape[0]} samples")
    return result


def make_model(obs_dim: int, hidden: list[int], device: str, checkpoint: Path | None = None) -> ProstheticMLP:
    model = ProstheticMLP(obs_dim, action_dim=3, hidden=hidden).to(device)
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model


def collect_dagger_data(
    model: ProstheticMLP,
    teacher,
    steps: int,
    scenario_pool_name: str,
    mass_rand_pct: float,
    friction_rand_pct: float,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    env = make_go1_env(render=False)
    pool = scenario_pool(scenario_pool_name)

    obs_list = []
    action_list = []
    desired_vel_list = []
    scenario_name_list = []
    scenario_id_list = []
    slope_list = []

    collected = 0
    episodes = 0
    while collected < steps:
        spec = sample_scenario(rng, scenario_pool_name)
        spec_idx = next(i for i, pool_spec in enumerate(pool) if pool_spec.name == spec.name)
        apply_scenario(env, spec, rng, mass_rand_pct=mass_rand_pct, friction_rand_pct=friction_rand_pct)
        obs, _ = env.reset(seed=seed + episodes)
        episodes += 1

        while collected < steps:
            teacher_action, _ = teacher.predict(obs, deterministic=True)
            obs_3leg, teacher_leg3 = split_obs_and_action(obs, teacher_action)

            with torch.no_grad():
                pred = model(torch.from_numpy(obs_3leg).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

            combined = teacher_action.copy()
            combined[GO1_LEG3_JOINT_IX] = pred

            obs_list.append(obs_3leg.astype(np.float32))
            action_list.append(teacher_leg3.astype(np.float32))
            desired_vel_list.append(np.array(spec.desired_velocity, dtype=np.float32))
            scenario_name_list.append(spec.name)
            scenario_id_list.append(spec_idx)
            slope_list.append(float(spec.slope_pitch_deg))

            obs, _, terminated, truncated, _ = env.step(combined)
            collected += 1
            if terminated or truncated:
                break

    env.close()
    print(f"Collected {collected} DAgger samples across {episodes} episodes")
    return {
        "obs_3leg": np.array(obs_list, dtype=np.float32),
        "action_leg3": np.array(action_list, dtype=np.float32),
        "desired_velocity": np.array(desired_vel_list, dtype=np.float32),
        "scenario_name": np.array(scenario_name_list),
        "scenario_id": np.array(scenario_id_list, dtype=np.int32),
        "slope_pitch_deg": np.array(slope_list, dtype=np.float32),
    }


def concat_datasets(base: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    keys = sorted(set(base.keys()) | set(extra.keys()))
    for key in keys:
        if key in base and key in extra:
            result[key] = np.concatenate([base[key], extra[key]], axis=0)
        elif key in base:
            result[key] = base[key]
        else:
            result[key] = extra[key]
    return result


def train_il(
    data_path: Path = DEFAULT_DATA,
    hidden: list[int] = [256, 256, 128],
    lr: float = 3e-4,
    batch_size: int = 2048,
    epochs: int = 18,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    patience: int = 6,
    dagger_iterations: int = 3,
    dagger_steps_per_iter: int = 120_000,
    scenario_pool_name: str = "all_train",
    mass_rand_pct: float = 0.10,
    friction_rand_pct: float = 0.20,
    teacher_model_path: str | None = None,
    bootstrap_checkpoint: Path | None = None,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    base = load_npz_dataset(data_path)
    aggregate = dict(base)

    teacher = load_teacher(teacher_model_path)
    iteration_logs = []
    checkpoint = bootstrap_checkpoint if bootstrap_checkpoint and bootstrap_checkpoint.exists() else None

    for iteration in range(dagger_iterations + 1):
        print(f"\n=== IL iteration {iteration}/{dagger_iterations} ===")
        X = torch.from_numpy(aggregate["obs_3leg"])
        y = torch.from_numpy(aggregate["action_leg3"])
        train_ds, val_ds = split_dataset(TensorDataset(X, y), val_fraction=0.1)

        initial_state = None
        if checkpoint and checkpoint.exists():
            initial_state = torch.load(checkpoint, map_location=device, weights_only=True)

        model, train_log = fit_model(
            train_ds=train_ds,
            val_ds=val_ds,
            save_dir=SAVE_DIR,
            hidden=hidden,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            num_workers=num_workers,
            patience=patience,
            initial_state_dict=initial_state,
            onnx_name="prosthetic_il.onnx",
        )
        checkpoint = SAVE_DIR / "best_model.pt"
        iteration_logs.append(
            {
                "iteration": iteration,
                "dataset_size": int(X.shape[0]),
                "best_val_loss": float(train_log["best_val_loss"]),
            }
        )

        if iteration == dagger_iterations:
            break

        student = make_model(X.shape[1], hidden, device, checkpoint=checkpoint)
        extra = collect_dagger_data(
            student,
            teacher,
            steps=dagger_steps_per_iter,
            scenario_pool_name=scenario_pool_name,
            mass_rand_pct=mass_rand_pct,
            friction_rand_pct=friction_rand_pct,
            seed=1234 + iteration,
            device=device,
        )
        aggregate = concat_datasets(aggregate, extra)
        np.savez_compressed(SAVE_DIR / "aggregated_rollouts.npz", **aggregate)

    metadata = {
        "base_data_path": str(data_path),
        "scenario_pool": scenario_pool_name,
        "dagger_iterations": dagger_iterations,
        "dagger_steps_per_iter": dagger_steps_per_iter,
        "mass_rand_pct": mass_rand_pct,
        "friction_rand_pct": friction_rand_pct,
        "total_samples": int(aggregate["obs_3leg"].shape[0]),
        "iterations": iteration_logs,
    }
    (SAVE_DIR / "dagger_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved IL artifacts to {SAVE_DIR}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-steps-per-iter", type=int, default=120000)
    parser.add_argument("--scenario-pool", type=str, default="all_train")
    parser.add_argument("--bootstrap", type=str, default="models/supervised_prosthetic/best_model.pt")
    args = parser.parse_args()

    kwargs = dict(
        hidden=[256, 256, 128],
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        patience=args.patience,
        dagger_iterations=args.dagger_iterations,
        dagger_steps_per_iter=args.dagger_steps_per_iter,
        scenario_pool_name=args.scenario_pool,
        bootstrap_checkpoint=(Path(args.bootstrap) if args.bootstrap and Path(args.bootstrap).is_absolute() else (REPO / args.bootstrap) if args.bootstrap else None),
    )
    if args.data:
        kwargs["data_path"] = Path(args.data)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kwargs.update(
            {
                "data_path": Path(cfg.get("data_path", kwargs.get("data_path", DEFAULT_DATA))),
                "hidden": cfg.get("hidden", kwargs["hidden"]),
                "lr": cfg.get("lr", kwargs["lr"]),
                "batch_size": cfg.get("batch_size", kwargs["batch_size"]),
                "epochs": cfg.get("epochs", kwargs["epochs"]),
                "num_workers": cfg.get("num_workers", kwargs["num_workers"]),
                "patience": cfg.get("patience", kwargs["patience"]),
                "scenario_pool_name": cfg.get("scenario_pool", kwargs["scenario_pool_name"]),
                "dagger_iterations": cfg.get("dagger_iterations", kwargs["dagger_iterations"]),
                "dagger_steps_per_iter": cfg.get("dagger_steps_per_iter", kwargs["dagger_steps_per_iter"]),
                "mass_rand_pct": cfg.get("data_gen_mass_rand", 0.10),
                "friction_rand_pct": cfg.get("data_gen_friction_rand", 0.20),
                "teacher_model_path": cfg.get("teacher_model_path"),
            }
        )

    train_il(**kwargs)


if __name__ == "__main__":
    main()
