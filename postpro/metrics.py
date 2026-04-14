"""
Derived metrics from raw training logs.

Given a RunData, computes higher-level diagnostics that tell you
*what actually happened* beyond raw reward curves:

  - Convergence speed / sample efficiency
  - Training stability (variance, plateaus, crashes)
  - Prosthetic leg behaviour (symmetry with intact legs)
  - Reward decomposition
  - Policy entropy / exploration trajectory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from postpro.load_logs import RunData, ScalarSeries


@dataclass
class ConvergenceMetrics:
    total_timesteps: int = 0
    final_reward: float = float("nan")
    best_reward: float = float("nan")
    best_step: int = 0
    steps_to_90pct: Optional[int] = None  # steps to reach 90% of best
    steps_to_first_positive: Optional[int] = None
    reward_at_10pct: float = float("nan")  # reward at 10% of training
    reward_at_50pct: float = float("nan")
    reward_at_90pct: float = float("nan")
    final_over_best: float = float("nan")  # 1.0 = no regression


@dataclass
class StabilityMetrics:
    reward_std_last_10pct: float = float("nan")
    max_reward_drop: float = float("nan")  # biggest single regression
    drop_step: int = 0
    n_plateaus: int = 0  # regions with <1% improvement over 10% of training
    n_crashes: int = 0   # reward drops > 20% of running max
    is_monotonic: bool = False
    coefficient_of_variation: float = float("nan")


@dataclass
class LegSymmetryMetrics:
    final_leg3_ratio: float = float("nan")  # leg3/others at end
    mean_leg3_ratio: float = float("nan")
    ratio_converged: bool = False  # ratio within [0.8, 1.2] in last 10%
    leg3_mean_action: float = float("nan")
    others_mean_action: float = float("nan")
    leg3_std_action: float = float("nan")
    others_std_action: float = float("nan")


@dataclass
class PolicyDynamics:
    final_entropy: float = float("nan")
    entropy_decay_rate: float = float("nan")  # negative = expected
    final_kl: float = float("nan")
    mean_clip_fraction: float = float("nan")
    final_explained_variance: float = float("nan")
    final_value_loss: float = float("nan")
    learning_rate_final: float = float("nan")


@dataclass
class DerivedMetrics:
    run_name: str
    run_type: str
    convergence: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    leg_symmetry: Optional[LegSymmetryMetrics] = None
    policy_dynamics: PolicyDynamics = field(default_factory=PolicyDynamics)
    eval_best_reward: float = float("nan")
    eval_best_step: int = 0
    eval_reward_std: float = float("nan")


def _percentile_step(steps: np.ndarray, values: np.ndarray, frac: float) -> float:
    idx = int(len(steps) * frac)
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def compute_convergence(reward_series: ScalarSeries) -> ConvergenceMetrics:
    m = ConvergenceMetrics()
    if len(reward_series.steps) == 0:
        return m

    steps, vals = reward_series.steps, reward_series.values
    m.total_timesteps = int(steps[-1])
    m.final_reward = float(vals[-1])
    m.best_reward = float(np.max(vals))
    m.best_step = int(steps[np.argmax(vals)])

    m.reward_at_10pct = _percentile_step(steps, vals, 0.1)
    m.reward_at_50pct = _percentile_step(steps, vals, 0.5)
    m.reward_at_90pct = _percentile_step(steps, vals, 0.9)
    m.final_over_best = float(vals[-1] / (m.best_reward + 1e-8))

    threshold_90 = m.best_reward * 0.9
    above = np.where(vals >= threshold_90)[0]
    if len(above) > 0:
        m.steps_to_90pct = int(steps[above[0]])

    positive = np.where(vals > 0)[0]
    if len(positive) > 0:
        m.steps_to_first_positive = int(steps[positive[0]])

    return m


def compute_stability(reward_series: ScalarSeries) -> StabilityMetrics:
    m = StabilityMetrics()
    if len(reward_series.values) < 5:
        return m

    vals = reward_series.values
    n = len(vals)
    tail = vals[int(n * 0.9):]
    m.reward_std_last_10pct = float(np.std(tail))
    m.coefficient_of_variation = float(np.std(tail) / (np.abs(np.mean(tail)) + 1e-8))

    running_max = np.maximum.accumulate(vals)
    drops = running_max - vals
    m.max_reward_drop = float(np.max(drops))
    m.drop_step = int(reward_series.steps[np.argmax(drops)])

    relative_drops = drops / (np.abs(running_max) + 1e-8)
    m.n_crashes = int(np.sum(relative_drops > 0.2))
    m.is_monotonic = bool(np.all(np.diff(vals) >= -1e-6))

    window = max(n // 10, 2)
    plateau_count = 0
    for i in range(0, n - window, window):
        chunk = vals[i:i + window]
        if len(chunk) < 2:
            continue
        improvement = (chunk[-1] - chunk[0]) / (np.abs(chunk[0]) + 1e-8)
        if abs(improvement) < 0.01:
            plateau_count += 1
    m.n_plateaus = plateau_count

    return m


def compute_leg_symmetry(run: RunData) -> Optional[LegSymmetryMetrics]:
    ratio_key = "leg3_vs_others_action_ratio"
    if ratio_key not in run.scalars:
        return None

    m = LegSymmetryMetrics()
    ratio = run.scalars[ratio_key]
    m.final_leg3_ratio = ratio.final
    m.mean_leg3_ratio = ratio.mean

    tail = ratio.values[int(len(ratio.values) * 0.9):]
    if len(tail) > 0:
        m.ratio_converged = bool(np.all((tail >= 0.8) & (tail <= 1.2)))

    if "leg_3_action_mean_abs" in run.scalars:
        m.leg3_mean_action = run.scalars["leg_3_action_mean_abs"].final

    others = []
    for i in range(3):
        key = f"leg_{i}_action_mean_abs"
        if key in run.scalars:
            others.append(run.scalars[key].final)
    if others:
        m.others_mean_action = float(np.mean(others))

    if "leg_3_action_std_abs" in run.scalars:
        m.leg3_std_action = run.scalars["leg_3_action_std_abs"].final

    others_std = []
    for i in range(3):
        key = f"leg_{i}_action_std_abs"
        if key in run.scalars:
            others_std.append(run.scalars[key].final)
    if others_std:
        m.others_std_action = float(np.mean(others_std))

    return m


def compute_policy_dynamics(run: RunData) -> PolicyDynamics:
    m = PolicyDynamics()
    tag_map = {
        "train/entropy_loss": "final_entropy",
        "train/approx_kl": "final_kl",
        "train/clip_fraction": "mean_clip_fraction",
        "train/explained_variance": "final_explained_variance",
        "train/value_loss": "final_value_loss",
        "train/learning_rate": "learning_rate_final",
    }
    for tag, attr in tag_map.items():
        if tag in run.scalars and len(run.scalars[tag].values) > 0:
            if "mean" in attr:
                setattr(m, attr, run.scalars[tag].mean)
            else:
                setattr(m, attr, run.scalars[tag].final)

    if "train/entropy_loss" in run.scalars:
        ent = run.scalars["train/entropy_loss"]
        if len(ent.values) > 10:
            first_q = ent.values[:len(ent.values) // 4]
            last_q = ent.values[-len(ent.values) // 4:]
            if len(first_q) > 0 and len(last_q) > 0:
                m.entropy_decay_rate = float(np.mean(last_q) - np.mean(first_q))

    return m


def compute_derived_metrics(run: RunData) -> DerivedMetrics:
    dm = DerivedMetrics(run_name=run.name, run_type=run.run_type)

    reward_key = "rollout/ep_rew_mean"
    if reward_key in run.scalars:
        dm.convergence = compute_convergence(run.scalars[reward_key])
        dm.stability = compute_stability(run.scalars[reward_key])

    dm.leg_symmetry = compute_leg_symmetry(run)
    dm.policy_dynamics = compute_policy_dynamics(run)

    if run.eval_log is not None:
        dm.eval_best_reward = run.eval_log.best_mean_reward
        dm.eval_best_step = run.eval_log.best_timestep
        dm.eval_reward_std = float(np.mean(run.eval_log.std_rewards))

    return dm
