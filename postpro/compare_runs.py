"""
Cross-run comparison: side-by-side analysis of different training approaches.

Answers questions like:
  - Did AMP produce better gaits than vanilla PPO?
  - How does prosthetic RL compare to supervised distillation?
  - Which seed/config was best?
  - Where did each approach plateau?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from postpro.load_logs import RunData
from postpro.metrics import DerivedMetrics


def _safe_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise SystemExit("pip install matplotlib")


def _fmt(v, p=3):
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Comparison table (text)
# ---------------------------------------------------------------------------

def comparison_table(metrics: list[DerivedMetrics]) -> str:
    if not metrics:
        return "No runs to compare."

    header = f"{'Run':<25} {'Type':<15} {'Best Rew':>10} {'Final Rew':>10} {'Fin/Best':>8} {'Steps90%':>10} {'Crashes':>7} {'Leg3 Ratio':>10}"
    sep = "-" * len(header)
    rows = [sep, header, sep]

    for dm in sorted(metrics, key=lambda d: -d.convergence.best_reward if not np.isnan(d.convergence.best_reward) else float("-inf")):
        c = dm.convergence
        s = dm.stability
        leg_r = _fmt(dm.leg_symmetry.final_leg3_ratio, 2) if dm.leg_symmetry else "n/a"
        rows.append(
            f"{dm.run_name:<25} {dm.run_type:<15} {_fmt(c.best_reward):>10} {_fmt(c.final_reward):>10} "
            f"{_fmt(c.final_over_best, 2):>8} {str(c.steps_to_90pct or 'n/a'):>10} {s.n_crashes:>7} {leg_r:>10}"
        )
    rows.append(sep)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Sample efficiency comparison
# ---------------------------------------------------------------------------

def plot_sample_efficiency(
    runs: list[RunData],
    metrics: list[DerivedMetrics],
    out_dir: Path,
    filename: str = "sample_efficiency.png",
):
    plt = _safe_import_plt()

    reward_runs = [(r, m) for r, m in zip(runs, metrics) if "rollout/ep_rew_mean" in r.scalars]
    if len(reward_runs) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(reward_runs)))

    # Left: normalized reward curves (0=start, 1=best achieved by any run)
    global_best = max(m.convergence.best_reward for _, m in reward_runs if not np.isnan(m.convergence.best_reward))
    global_worst = min(
        r.scalars["rollout/ep_rew_mean"].values[0]
        for r, _ in reward_runs if len(r.scalars["rollout/ep_rew_mean"].values) > 0
    )
    span = global_best - global_worst if global_best != global_worst else 1.0

    for i, (run, dm) in enumerate(reward_runs):
        s = run.scalars["rollout/ep_rew_mean"]
        normalized = (s.values - global_worst) / span
        from postpro.report import _smooth
        smoothed = _smooth(normalized, 20)
        x_smooth = s.steps[:len(smoothed)]
        axes[0].plot(x_smooth, smoothed, label=run.name, color=colors[i], linewidth=2)

    axes[0].axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="90% of best")
    axes[0].set_title("Normalized Reward (0 = worst start, 1 = global best)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Normalized Reward")
    axes[0].legend(fontsize=8)

    # Right: bar chart of key metrics
    names = [dm.run_name for _, dm in reward_runs]
    best_rewards = [dm.convergence.best_reward for _, dm in reward_runs]
    final_rewards = [dm.convergence.final_reward for _, dm in reward_runs]

    x = np.arange(len(names))
    w = 0.35
    axes[1].bar(x - w / 2, best_rewards, w, label="Best Reward", color="#4CAF50", alpha=0.8)
    axes[1].bar(x + w / 2, final_rewards, w, label="Final Reward", color="#2196F3", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Best vs Final Reward")
    axes[1].legend(fontsize=8)

    fig.suptitle("Sample Efficiency Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


# ---------------------------------------------------------------------------
# Approach comparison radar chart
# ---------------------------------------------------------------------------

def plot_radar_comparison(
    metrics: list[DerivedMetrics],
    out_dir: Path,
    filename: str = "radar_comparison.png",
):
    plt = _safe_import_plt()

    scoreable = [dm for dm in metrics if not np.isnan(dm.convergence.best_reward)]
    if len(scoreable) < 2:
        return

    categories = [
        "Peak Reward",
        "Stability",
        "Sample Eff.",
        "Leg Symmetry",
        "Final / Best",
    ]

    def _score(dm: DerivedMetrics) -> list[float]:
        scores = []
        scores.append(dm.convergence.best_reward)

        stability_score = 1.0 / (1.0 + dm.stability.n_crashes + dm.stability.n_plateaus)
        scores.append(stability_score)

        if dm.convergence.steps_to_90pct and dm.convergence.total_timesteps > 0:
            scores.append(1.0 - dm.convergence.steps_to_90pct / dm.convergence.total_timesteps)
        else:
            scores.append(0.5)

        if dm.leg_symmetry and not np.isnan(dm.leg_symmetry.final_leg3_ratio):
            scores.append(1.0 - abs(1.0 - dm.leg_symmetry.final_leg3_ratio))
        else:
            scores.append(0.5)

        scores.append(dm.convergence.final_over_best if not np.isnan(dm.convergence.final_over_best) else 0.5)
        return scores

    all_scores = [_score(dm) for dm in scoreable]

    # Normalize each category to [0, 1]
    all_scores_arr = np.array(all_scores)
    mins = all_scores_arr.min(axis=0)
    maxs = all_scores_arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (all_scores_arr - mins) / ranges

    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(scoreable)))

    for i, (dm, norm) in enumerate(zip(scoreable, normalized)):
        values = norm.tolist() + [norm[0]]
        ax.plot(angles, values, "o-", linewidth=2, label=dm.run_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.set_title("Run Comparison", fontsize=13, pad=20)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


# ---------------------------------------------------------------------------
# Master comparison
# ---------------------------------------------------------------------------

def compare_all(
    runs: list[RunData],
    metrics: list[DerivedMetrics],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Cross-Run Comparison ---\n")

    table = comparison_table(metrics)
    print(table)
    (out_dir / "comparison_table.txt").write_text(table)

    plot_sample_efficiency(runs, metrics, out_dir)
    plot_radar_comparison(metrics, out_dir)
