"""
Bounded end-to-end experiment runner for the Bark prosthetic Go1 stack.

Goals:
  1. Generate richer teacher data across multiple commands / speeds / slopes.
  2. Retrain the supervised student with GPU-heavy settings where available.
  3. Train an explicit IL student with DAgger-style aggregation.
  4. Retrain the RL student with parallel MuJoCo environments.
  5. Recompute comparison plots and reports.

Default settings are chosen to finish comfortably below a 2-hour budget on a typical
desktop while still producing meaningfully larger data and fresher models.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO))

from train.generate_teacher_data import generate  # noqa: E402
from train.train_supervised import train as train_supervised  # noqa: E402
from train.train_il import train_il  # noqa: E402


def _run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--data-steps", type=int, default=1_000_000)
    parser.add_argument("--scenario-pool", type=str, default="all_train")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--mass-rand", type=float, default=0.10)
    parser.add_argument("--friction-rand", type=float, default=0.20)
    parser.add_argument("--supervised-epochs", type=int, default=24)
    parser.add_argument("--supervised-batch-size", type=int, default=2048)
    parser.add_argument("--supervised-lr", type=float, default=3e-4)
    parser.add_argument("--il-epochs", type=int, default=18)
    parser.add_argument("--il-batch-size", type=int, default=2048)
    parser.add_argument("--il-dagger-iterations", type=int, default=3)
    parser.add_argument("--il-dagger-steps", type=int, default=120_000)
    parser.add_argument("--rl-timesteps", type=int, default=500_000)
    parser.add_argument("--rl-n-envs", type=int, default=4)
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--compare-episodes", type=int, default=24)
    parser.add_argument("--skip-postpro", action="store_true")
    args = parser.parse_args()

    total_start = time.perf_counter()

    print("=" * 72)
    print("BARK FULL EXPERIMENT")
    print("=" * 72)
    print(f"repo: {REPO}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"data_steps={args.data_steps}, supervised_epochs={args.supervised_epochs}, rl_timesteps={args.rl_timesteps}")

    stage_start = time.perf_counter()
    generate(
        model_path=args.teacher,
        total_steps=args.data_steps,
        obs_noise_std=args.noise,
        mass_rand_pct=args.mass_rand,
        friction_rand_pct=args.friction_rand,
        scenario_pool_name=args.scenario_pool,
    )
    print(f"[done] teacher data generation in {(time.perf_counter() - stage_start) / 60:.1f} min")

    stage_start = time.perf_counter()
    supervised_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_supervised(
        lr=args.supervised_lr,
        batch_size=args.supervised_batch_size,
        epochs=args.supervised_epochs,
        device=supervised_device,
        num_workers=4,
        patience=6,
    )
    print(f"[done] supervised training in {(time.perf_counter() - stage_start) / 60:.1f} min")

    stage_start = time.perf_counter()
    train_il(
        lr=args.supervised_lr,
        batch_size=args.il_batch_size,
        epochs=args.il_epochs,
        device=supervised_device,
        num_workers=4,
        patience=6,
        dagger_iterations=args.il_dagger_iterations,
        dagger_steps_per_iter=args.il_dagger_steps,
        scenario_pool_name=args.scenario_pool,
        mass_rand_pct=args.mass_rand,
        friction_rand_pct=args.friction_rand,
        teacher_model_path=args.teacher,
        bootstrap_checkpoint=REPO / "models" / "supervised_prosthetic" / "best_model.pt",
    )
    print(f"[done] IL training in {(time.perf_counter() - stage_start) / 60:.1f} min")

    stage_start = time.perf_counter()
    rl_cmd = [
        sys.executable,
        "train/train_prosthetic_rl.py",
        "--config",
        "configs/prosthetic_rl_go1.yaml",
        "--timesteps",
        str(args.rl_timesteps),
        "--n-envs",
        str(args.rl_n_envs),
        "--device",
        args.rl_device,
        "--obs-noise",
        str(args.noise),
    ]
    if args.teacher:
        rl_cmd.extend(["--teacher", args.teacher])
    _run(rl_cmd)
    print(f"[done] RL training in {(time.perf_counter() - stage_start) / 60:.1f} min")

    stage_start = time.perf_counter()
    compare_cmd = [
        sys.executable,
        "evaluate/compare.py",
        "--episodes",
        str(args.compare_episodes),
    ]
    if args.teacher:
        compare_cmd.extend(["--teacher", args.teacher])
    compare_cmd.extend(["--scenario-pool", args.scenario_pool])
    _run(compare_cmd)
    _run([sys.executable, "-m", "postpro.render_students", "--steps", "500", "--fps", "30"])
    if not args.skip_postpro:
        _run([sys.executable, "-m", "postpro.run_all"])
    print(f"[done] plots and reports in {(time.perf_counter() - stage_start) / 60:.1f} min")

    total_minutes = (time.perf_counter() - total_start) / 60
    print("=" * 72)
    print(f"Experiment complete in {total_minutes:.1f} min")
    print("Generated artifacts:")
    print("  - data/teacher_rollouts.npz")
    print("  - models/supervised_prosthetic/*")
    print("  - models/imitation_prosthetic/*")
    print("  - models/prosthetic_rl/*")
    print("  - reports/student_comparison.png")
    print("  - reports/leg3_action_traces.png")
    print("  - reports/leg3_tracking_error.png")
    print("  - reports/leg3_reward_comparison.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
