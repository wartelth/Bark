"""
Auto-generate a full training report: text summary + publication-quality figures.

Produces:
  reports/
    summary.txt          - human-readable narrative of what happened
    reward_curves.png    - reward over training (all runs overlaid)
    eval_progression.png - eval callback results with confidence bands
    leg_symmetry.png     - per-leg action magnitudes + ratio
    policy_internals.png - entropy, KL, clip fraction, value loss
    stability.png        - reward volatility and crash markers
"""
from __future__ import annotations

import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from postpro.load_logs import RunData, ScalarSeries, EvalLog
from postpro.metrics import DerivedMetrics


def _safe_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        })
        return plt
    except ImportError:
        raise SystemExit("pip install matplotlib  (needed for report generation)")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def _fmt(v, precision=4) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        return f"{v:.{precision}f}"
    return str(v)


def generate_text_summary(
    runs: list[RunData],
    metrics: list[DerivedMetrics],
    out_path: Path,
):
    lines = [
        "=" * 72,
        f"  BARK Training Post-Processing Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 72,
        "",
    ]

    for run, dm in zip(runs, metrics):
        lines.append(f"--- {run.name} ({run.run_type}) ---")
        lines.append(f"  Source: {run.source_dir}")
        lines.append(f"  Scalars found: {len(run.scalars)} tags")
        lines.append(f"  Eval log: {'yes' if run.eval_log else 'no'}")
        lines.append("")

        c = dm.convergence
        lines.append("  Convergence:")
        lines.append(f"    Total timesteps:       {c.total_timesteps:,}")
        lines.append(f"    Final reward:           {_fmt(c.final_reward)}")
        lines.append(f"    Best reward:            {_fmt(c.best_reward)} (step {c.best_step:,})")
        lines.append(f"    Final / best ratio:     {_fmt(c.final_over_best)}")
        lines.append(f"    Steps to 90% of best:   {c.steps_to_90pct or 'n/a'}")
        lines.append(f"    Reward @ 10%/50%/90%:   {_fmt(c.reward_at_10pct)} / {_fmt(c.reward_at_50pct)} / {_fmt(c.reward_at_90pct)}")
        lines.append("")

        s = dm.stability
        lines.append("  Stability:")
        lines.append(f"    Reward std (last 10%):  {_fmt(s.reward_std_last_10pct)}")
        lines.append(f"    Coeff of variation:     {_fmt(s.coefficient_of_variation)}")
        lines.append(f"    Max single drop:        {_fmt(s.max_reward_drop)} (step {s.drop_step:,})")
        lines.append(f"    Crash events (>20%):    {s.n_crashes}")
        lines.append(f"    Plateau regions:        {s.n_plateaus}")
        lines.append("")

        if dm.leg_symmetry:
            ls = dm.leg_symmetry
            lines.append("  Leg Symmetry:")
            lines.append(f"    Final leg3/others:      {_fmt(ls.final_leg3_ratio)}")
            lines.append(f"    Mean leg3/others:        {_fmt(ls.mean_leg3_ratio)}")
            lines.append(f"    Converged to [0.8,1.2]:  {'yes' if ls.ratio_converged else 'no'}")
            lines.append(f"    Leg3 mean |action|:      {_fmt(ls.leg3_mean_action)}")
            lines.append(f"    Others mean |action|:    {_fmt(ls.others_mean_action)}")
            lines.append("")

        p = dm.policy_dynamics
        lines.append("  Policy Dynamics:")
        lines.append(f"    Final entropy:          {_fmt(p.final_entropy)}")
        lines.append(f"    Entropy change:         {_fmt(p.entropy_decay_rate)}")
        lines.append(f"    Final approx KL:        {_fmt(p.final_kl)}")
        lines.append(f"    Mean clip fraction:     {_fmt(p.mean_clip_fraction)}")
        lines.append(f"    Final expl. variance:   {_fmt(p.final_explained_variance)}")
        lines.append(f"    Final value loss:       {_fmt(p.final_value_loss)}")
        lines.append("")

        if not np.isnan(dm.eval_best_reward):
            lines.append(f"  Eval Callback:")
            lines.append(f"    Best eval reward:       {_fmt(dm.eval_best_reward)} (step {dm.eval_best_step:,})")
            lines.append(f"    Mean eval std:          {_fmt(dm.eval_reward_std)}")
            lines.append("")

        lines.append("")

    # Narrative verdict
    lines.append("=" * 72)
    lines.append("  VERDICT")
    lines.append("=" * 72)
    for dm in metrics:
        c = dm.convergence
        s = dm.stability
        verdict_parts = []
        if c.final_over_best > 0.95:
            verdict_parts.append("converged and held")
        elif c.final_over_best > 0.8:
            verdict_parts.append("converged with slight regression")
        else:
            verdict_parts.append("significant regression from peak")
        if s.n_crashes > 3:
            verdict_parts.append("unstable (multiple crash events)")
        elif s.n_crashes > 0:
            verdict_parts.append(f"minor instability ({s.n_crashes} crash event(s))")
        else:
            verdict_parts.append("stable")
        if dm.leg_symmetry and dm.leg_symmetry.ratio_converged:
            verdict_parts.append("prosthetic leg converged to natural gait")
        elif dm.leg_symmetry:
            verdict_parts.append(f"prosthetic leg ratio={_fmt(dm.leg_symmetry.final_leg3_ratio, 2)} (target ~1.0)")
        lines.append(f"  [{dm.run_name}] {'; '.join(verdict_parts)}")
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_curves(
    runs: list[RunData],
    out_dir: Path,
    filename: str = "reward_curves.png",
):
    plt = _safe_import_plt()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(runs), 1)))

    for i, run in enumerate(runs):
        tag = "rollout/ep_rew_mean"
        if tag not in run.scalars:
            continue
        s = run.scalars[tag]
        axes[0].plot(s.steps, s.values, alpha=0.25, color=colors[i])
        smoothed = _smooth(s.values)
        x_smooth = s.steps[:len(smoothed)]
        axes[0].plot(x_smooth, smoothed, label=run.name, color=colors[i], linewidth=2)

        eval_tag = "eval/mean_reward"
        if eval_tag in run.scalars:
            es = run.scalars[eval_tag]
            axes[0].plot(es.steps, es.values, "o--", color=colors[i], markersize=3, alpha=0.6,
                         label=f"{run.name} (eval)")

    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Reward Progression")
    axes[0].legend(fontsize=8, loc="lower right")

    for i, run in enumerate(runs):
        tag = "rollout/ep_len_mean"
        if tag not in run.scalars:
            continue
        s = run.scalars[tag]
        smoothed = _smooth(s.values)
        x_smooth = s.steps[:len(smoothed)]
        axes[1].plot(x_smooth, smoothed, label=run.name, color=colors[i], linewidth=2)

    axes[1].set_ylabel("Episode Length")
    axes[1].set_xlabel("Timestep")
    axes[1].set_title("Episode Survival")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_eval_progression(
    runs: list[RunData],
    out_dir: Path,
    filename: str = "eval_progression.png",
):
    plt = _safe_import_plt()
    has_eval = [r for r in runs if r.eval_log is not None]
    if not has_eval:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(has_eval), 1)))

    for i, run in enumerate(has_eval):
        ev = run.eval_log
        means = ev.mean_rewards
        stds = ev.std_rewards
        ax.plot(ev.timesteps, means, label=run.name, color=colors[i], linewidth=2)
        ax.fill_between(ev.timesteps, means - stds, means + stds, color=colors[i], alpha=0.15)
        best_idx = np.argmax(means)
        ax.annotate(
            f"best: {means[best_idx]:.1f}",
            xy=(ev.timesteps[best_idx], means[best_idx]),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color=colors[i]),
            xytext=(ev.timesteps[best_idx], means[best_idx] + stds[best_idx] + 5),
            color=colors[i],
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Eval Reward")
    ax.set_title("Evaluation Callback Progression (mean +/- std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_leg_symmetry(
    runs: list[RunData],
    out_dir: Path,
    filename: str = "leg_symmetry.png",
):
    plt = _safe_import_plt()

    leg_runs = [r for r in runs if "leg_3_action_mean_abs" in r.scalars]
    if not leg_runs:
        return

    fig, axes = plt.subplots(len(leg_runs), 2, figsize=(14, 5 * len(leg_runs)), squeeze=False)

    leg_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    leg_labels = ["Leg 0", "Leg 1", "Leg 2", "Leg 3 (prosthetic)"]

    for row, run in enumerate(leg_runs):
        ax_mag, ax_ratio = axes[row]

        for leg_idx in range(4):
            tag = f"leg_{leg_idx}_action_mean_abs"
            if tag not in run.scalars:
                continue
            s = run.scalars[tag]
            smoothed = _smooth(s.values, 10)
            x_smooth = s.steps[:len(smoothed)]
            lw = 2.5 if leg_idx == 3 else 1.5
            ax_mag.plot(x_smooth, smoothed, color=leg_colors[leg_idx],
                        label=leg_labels[leg_idx], linewidth=lw)

        ax_mag.set_ylabel("Mean |action|")
        ax_mag.set_title(f"{run.name}: Per-Leg Action Magnitude")
        ax_mag.legend(fontsize=8)

        ratio_tag = "leg3_vs_others_action_ratio"
        if ratio_tag in run.scalars:
            s = run.scalars[ratio_tag]
            smoothed = _smooth(s.values, 10)
            x_smooth = s.steps[:len(smoothed)]
            ax_ratio.plot(x_smooth, smoothed, color="#F44336", linewidth=2)
            ax_ratio.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="target = 1.0")
            ax_ratio.axhspan(0.8, 1.2, color="green", alpha=0.08, label="acceptable range")
            ax_ratio.set_ylabel("Leg3 / Others Ratio")
            ax_ratio.set_title(f"{run.name}: Prosthetic Convergence")
            ax_ratio.legend(fontsize=8)

        ax_mag.set_xlabel("Timestep")
        ax_ratio.set_xlabel("Timestep")

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_policy_internals(
    runs: list[RunData],
    out_dir: Path,
    filename: str = "policy_internals.png",
):
    plt = _safe_import_plt()

    tags = [
        ("train/entropy_loss", "Entropy Loss"),
        ("train/approx_kl", "Approx KL"),
        ("train/clip_fraction", "Clip Fraction"),
        ("train/value_loss", "Value Loss"),
        ("train/explained_variance", "Explained Variance"),
        ("train/loss", "Total Loss"),
    ]

    present_tags = []
    for tag, label in tags:
        if any(tag in r.scalars for r in runs):
            present_tags.append((tag, label))

    if not present_tags:
        return

    n_plots = len(present_tags)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(runs), 1)))

    for idx, (tag, label) in enumerate(present_tags):
        ax = axes[idx // cols][idx % cols]
        for i, run in enumerate(runs):
            if tag not in run.scalars:
                continue
            s = run.scalars[tag]
            smoothed = _smooth(s.values, 15)
            x_smooth = s.steps[:len(smoothed)]
            ax.plot(x_smooth, smoothed, label=run.name, color=colors[i], linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=7)

    for idx in range(len(present_tags), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Policy Training Internals", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_stability(
    runs: list[RunData],
    metrics: list[DerivedMetrics],
    out_dir: Path,
    filename: str = "stability.png",
):
    plt = _safe_import_plt()

    reward_runs = [(r, m) for r, m in zip(runs, metrics) if "rollout/ep_rew_mean" in r.scalars]
    if not reward_runs:
        return

    fig, axes = plt.subplots(len(reward_runs), 1, figsize=(12, 4 * len(reward_runs)), squeeze=False)

    for row, (run, dm) in enumerate(reward_runs):
        ax = axes[row][0]
        s = run.scalars["rollout/ep_rew_mean"]
        ax.plot(s.steps, s.values, alpha=0.3, color="steelblue", linewidth=0.8)
        smoothed = _smooth(s.values, 30)
        x_smooth = s.steps[:len(smoothed)]
        ax.plot(x_smooth, smoothed, color="steelblue", linewidth=2, label="reward (smoothed)")

        running_max = np.maximum.accumulate(s.values)
        ax.plot(s.steps, running_max, color="green", linewidth=1, linestyle="--", alpha=0.5, label="running max")

        drops = running_max - s.values
        relative_drops = drops / (np.abs(running_max) + 1e-8)
        crash_mask = relative_drops > 0.2
        if np.any(crash_mask):
            ax.scatter(s.steps[crash_mask], s.values[crash_mask], color="red", s=20,
                       zorder=5, label=f"crashes ({np.sum(crash_mask)})")

        if dm.stability.drop_step > 0:
            ax.axvline(dm.stability.drop_step, color="red", linestyle=":", alpha=0.4)

        ax.set_title(f"{run.name}: Reward Stability")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


# ---------------------------------------------------------------------------
# Master report generator
# ---------------------------------------------------------------------------

def generate_report(
    runs: list[RunData],
    metrics: list[DerivedMetrics],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating report in {out_dir}/\n")

    generate_text_summary(runs, metrics, out_dir / "summary.txt")
    print()

    plot_reward_curves(runs, out_dir)
    plot_eval_progression(runs, out_dir)
    plot_leg_symmetry(runs, out_dir)
    plot_policy_internals(runs, out_dir)
    plot_stability(runs, metrics, out_dir)

    print(f"\nReport complete: {out_dir}/")
