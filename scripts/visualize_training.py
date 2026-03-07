"""
Visualize training logs: reward curves and per-leg action metrics.
Answers: "Does the prosthetic leg (leg 3) train to behave like the observed legs?"

Usage:
  PYTHONPATH=. python scripts/visualize_training.py --logdir logs/tensorboard
  PYTHONPATH=. python scripts/visualize_training.py --logdir logs/tensorboard --run PPO_1 --out logs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_tb_scalars(
    logdir: Path,
    run_name: str | None,
    tags: list[str],
    *,
    all_runs: bool = False,
):
    """
    Load scalar series from TensorBoard event files.
    If all_runs is False: returns {tag: [(step, value), ...]} for one run.
    If all_runs is True: returns [(run_name, {tag: [(step, value), ...]}), ...] for every run.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        EventAccumulator = event_accumulator.EventAccumulator
        SCALARS = getattr(event_accumulator, "SCALARS", "scalars")
    except ImportError:
        raise SystemExit("Install tensorboard to use this script: pip install tensorboard")

    if run_name:
        run_dirs = [(run_name, logdir / run_name)]
    else:
        run_dirs = [(d.name, d) for d in sorted(logdir.iterdir()) if d.is_dir()]
    if not run_dirs:
        raise SystemExit(f"No run directories found under {logdir}")

    result_list: list[tuple[str, dict[str, list[tuple[int, float]]]]] = []
    for rname, run_dir in run_dirs:
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        event_path = max(event_files, key=lambda p: p.stat().st_mtime)
        size_guidance = {SCALARS: 0}
        ea = EventAccumulator(str(event_path), size_guidance=size_guidance)
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        out: dict[str, list[tuple[int, float]]] = {t: [] for t in tags}
        for tag in tags:
            if tag not in available:
                continue
            for e in ea.Scalars(tag):
                out[tag].append((e.step, e.value))
        for tag in out:
            out[tag] = sorted(out[tag], key=lambda x: x[0])
        result_list.append((rname, out))
        if not all_runs and run_name:
            break
    if all_runs:
        return result_list
    if not result_list:
        return {t: [] for t in tags}
    return result_list[0][1]


def main():
    parser = argparse.ArgumentParser(description="Plot training and per-leg metrics from TensorBoard logs")
    parser.add_argument("--logdir", type=str, default="logs/tensorboard", help="TensorBoard log directory")
    parser.add_argument("--run", type=str, default=None, help="Specific run folder (e.g. PPO_1); if not set, use first found")
    parser.add_argument("--all-runs", action="store_true", help="Plot all runs in logdir on the same figures (compare seeds/configs)")
    parser.add_argument("--out", type=str, default="logs/figures", help="Output directory for plots")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    logdir = repo / args.logdir
    out_dir = repo / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tags_reward = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "eval/mean_reward",
    ]
    tags_legs = [
        "leg_0_action_mean_abs",
        "leg_1_action_mean_abs",
        "leg_2_action_mean_abs",
        "leg_3_action_mean_abs",
        "leg3_vs_others_action_ratio",
    ]
    all_tags = tags_reward + tags_legs

    if args.all_runs:
        runs_data = load_tb_scalars(logdir, args.run, all_tags, all_runs=True)
        if not runs_data:
            raise SystemExit("No run data found.")
        # Single-dict view for compatibility: use first run only for "data" in single-run style
        data = runs_data[0][1] if runs_data else {}
    else:
        data = load_tb_scalars(logdir, args.run, all_tags)
        runs_data = [("run", data)] if data and any(data.get(t) for t in all_tags) else []
    if not runs_data:
        raise SystemExit("No run data found. Train with --tb_dir and run this script with --logdir pointing to that directory.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib: pip install matplotlib")

    run_colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs_data), 1)))

    # Figure 1: Reward and episode length (optionally multiple runs)
    fig1, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for run_idx, (rname, d) in enumerate(runs_data):
        prefix = f"{rname}: " if args.all_runs and len(runs_data) > 1 else ""
        if d.get("rollout/ep_rew_mean"):
            steps, vals = zip(*d["rollout/ep_rew_mean"])
            axes[0].plot(steps, vals, label=f"{prefix}Rollout reward", color=run_colors[run_idx % len(run_colors)], alpha=0.9)
        if d.get("eval/mean_reward"):
            steps, vals = zip(*d["eval/mean_reward"])
            axes[0].plot(steps, vals, label=f"{prefix}Eval reward", color=run_colors[run_idx % len(run_colors)], linestyle="--", alpha=0.8)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training reward")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for run_idx, (rname, d) in enumerate(runs_data):
        if d.get("rollout/ep_len_mean"):
            steps, vals = zip(*d["rollout/ep_len_mean"])
            axes[1].plot(steps, vals, color=run_colors[run_idx % len(run_colors)], label=rname if args.all_runs else "Episode length")
    axes[1].set_ylabel("Length")
    axes[1].set_xlabel("Step")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_dir / "training_reward.png", dpi=150)
    plt.close(fig1)
    print("Saved", out_dir / "training_reward.png")

    # Figure 2: Per-leg action magnitude — "Does leg 3 train like the others?"
    leg_tags = [f"leg_{i}_action_mean_abs" for i in range(4)]
    if any(data.get(t) for t in leg_tags):
        fig2, ax1 = plt.subplots(figsize=(10, 5))
        colors = ["C0", "C1", "C2", "C3"]
        labels = ["Leg 0", "Leg 1", "Leg 2", "Leg 3 (prosthetic)"]
        for tag, color, label in zip(leg_tags, colors, labels):
            if not data.get(tag):
                continue
            steps, vals = zip(*data[tag])
            ax1.plot(steps, vals, color=color, label=label, alpha=0.9)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Mean |action|")
        ax1.set_title("Per-leg action magnitude (prosthetic leg 3 should converge toward legs 0–2)")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        if data.get("leg3_vs_others_action_ratio"):
            steps, vals = zip(*data["leg3_vs_others_action_ratio"])
            ax2.plot(steps, vals, color="black", linestyle="--", alpha=0.7, label="Leg3 / others ratio")
            ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
            ax2.set_ylabel("Leg 3 vs others ratio")
            ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(out_dir / "per_leg_actions.png", dpi=150)
        plt.close(fig2)
        print("Saved", out_dir / "per_leg_actions.png")
    else:
        print("No per-leg metrics found; run training with LegMetricsCallback and --tb_dir to generate them.")

    print("Done. Open the PNGs to see if leg 3 trains like the missing one (similar mean |action| and ratio near 1).")


if __name__ == "__main__":
    main()
