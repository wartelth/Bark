"""
Unified log loader: discovers and parses every training artifact Bark produces.

Handles:
  - TensorBoard event files (RL, prosthetic RL, teacher)
  - SB3 evaluations.npz (EvalCallback outputs)
  - Supervised training loss (from model timestamps / manual CSV if present)
  - Monitor CSVs (episode-level stats from gymnasium.Monitor)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from postpro import (
    DEFAULT_TB_DIR,
    DEFAULT_EVAL_DIR,
    IMITATION_DIR,
    PROSTHETIC_RL_DIR,
    SUPERVISED_DIR,
    TEACHER_DIR,
    REPO_ROOT,
)


@dataclass
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray

    @property
    def final(self) -> float:
        return float(self.values[-1]) if len(self.values) > 0 else float("nan")

    @property
    def best(self) -> float:
        return float(np.max(self.values)) if len(self.values) > 0 else float("nan")

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if len(self.values) > 0 else float("nan")

    def window_mean(self, n: int = 50) -> np.ndarray:
        if len(self.values) < n:
            return self.values.copy()
        kernel = np.ones(n) / n
        return np.convolve(self.values, kernel, mode="valid")


@dataclass
class EvalLog:
    timesteps: np.ndarray
    results: np.ndarray  # (n_evals, n_episodes)
    ep_lengths: Optional[np.ndarray] = None

    @property
    def mean_rewards(self) -> np.ndarray:
        return self.results.mean(axis=1)

    @property
    def std_rewards(self) -> np.ndarray:
        return self.results.std(axis=1)

    @property
    def best_mean_reward(self) -> float:
        return float(np.max(self.mean_rewards))

    @property
    def best_timestep(self) -> int:
        return int(self.timesteps[np.argmax(self.mean_rewards)])


@dataclass
class RunData:
    name: str
    source_dir: Path
    run_type: str  # "rl", "prosthetic_rl", "teacher", "supervised", "il", "bc"
    scalars: dict[str, ScalarSeries] = field(default_factory=dict)
    eval_log: Optional[EvalLog] = None
    config: Optional[dict] = None
    monitor_episodes: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# TensorBoard loader
# ---------------------------------------------------------------------------

def _load_tb_event_file(event_path: Path, tags: list[str] | None = None) -> dict[str, ScalarSeries]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise SystemExit("pip install tensorboard  (needed for post-processing)")

    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    if tags is None:
        tags = available

    result = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        order = np.argsort(steps)
        result[tag] = ScalarSeries(tag=tag, steps=steps[order], values=values[order])
    return result


def load_tb_dir(tb_dir: Path, tags: list[str] | None = None) -> dict[str, ScalarSeries]:
    """Load scalars from the most recent event file under a TB directory tree."""
    event_files = sorted(tb_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return {}
    return _load_tb_event_file(event_files[-1], tags)


def load_all_tb_runs(tb_dir: Path, tags: list[str] | None = None) -> list[tuple[str, dict[str, ScalarSeries]]]:
    """Load every run under a TB directory (each sub-directory = one run)."""
    runs = []
    for d in sorted(tb_dir.iterdir()):
        if not d.is_dir():
            continue
        event_files = sorted(d.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if event_files:
            scalars = _load_tb_event_file(event_files[-1], tags)
            if scalars:
                runs.append((d.name, scalars))
    if not runs:
        scalars = load_tb_dir(tb_dir, tags)
        if scalars:
            runs.append((tb_dir.name, scalars))
    return runs


# ---------------------------------------------------------------------------
# evaluations.npz loader
# ---------------------------------------------------------------------------

def load_eval_npz(path: Path) -> Optional[EvalLog]:
    if not path.exists():
        return None
    data = np.load(path)
    timesteps = data.get("timesteps")
    results = data.get("results")
    if timesteps is None or results is None:
        return None
    ep_lengths = data.get("ep_lengths")
    return EvalLog(timesteps=timesteps, results=results, ep_lengths=ep_lengths)


# ---------------------------------------------------------------------------
# Monitor CSV loader
# ---------------------------------------------------------------------------

def load_monitor_csv(path: Path) -> Optional[np.ndarray]:
    """Parse a gymnasium Monitor CSV → structured array with r, l, t columns."""
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("r,l,t") or line.startswith('"r"'):
            header_idx = i
            break
    if header_idx is None:
        return None
    rows = []
    for line in lines[header_idx + 1:]:
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                rows.append((float(parts[0]), int(float(parts[1])), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        return None
    return np.array(rows, dtype=[("r", float), ("l", int), ("t", float)])


# ---------------------------------------------------------------------------
# Supervised training artifacts
# ---------------------------------------------------------------------------

def load_supervised_loss_log(model_dir: Path) -> Optional[dict]:
    """
    Attempt to reconstruct supervised training history.
    Checks for a JSON log or infers from model file timestamps.
    """
    log_path = model_dir / "training_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text())

    best = model_dir / "best_model.pt"
    final = model_dir / "final_model.pt"
    info: dict = {"has_best": best.exists(), "has_final": final.exists()}
    if best.exists():
        info["best_model_mtime"] = best.stat().st_mtime
        info["best_model_size_kb"] = best.stat().st_size / 1024
    if final.exists():
        info["final_model_mtime"] = final.stat().st_mtime
    onnx = model_dir / "prosthetic.onnx"
    if onnx.exists():
        info["has_onnx"] = True
        info["onnx_size_kb"] = onnx.stat().st_size / 1024
    return info if any(info.values()) else None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config_for_run(run_name: str) -> Optional[dict]:
    """Try to match a run name to a YAML config file."""
    import yaml

    configs_dir = REPO_ROOT / "configs"
    if not configs_dir.exists():
        return None

    for cfg_path in configs_dir.glob("*.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            rn = cfg.get("run_name", "")
            env_id = cfg.get("env_id", "")
            if run_name.lower() in str(cfg_path.stem).lower():
                return {**cfg, "_config_file": str(cfg_path)}
            if rn and run_name.lower() in rn.lower():
                return {**cfg, "_config_file": str(cfg_path)}
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Unified discovery
# ---------------------------------------------------------------------------

_KNOWN_TAGS = [
    "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "eval/mean_reward", "eval/mean_ep_length",
    "train/loss", "train/entropy_loss", "train/policy_gradient_loss",
    "train/value_loss", "train/approx_kl", "train/clip_fraction",
    "train/explained_variance", "train/learning_rate",
    "leg_0_action_mean_abs", "leg_1_action_mean_abs",
    "leg_2_action_mean_abs", "leg_3_action_mean_abs",
    "leg_0_action_std_abs", "leg_1_action_std_abs",
    "leg_2_action_std_abs", "leg_3_action_std_abs",
    "leg3_vs_others_action_ratio",
    "amp_d_loss",
]


def discover_runs(
    tb_dir: Path = DEFAULT_TB_DIR,
    include_prosthetic: bool = True,
    include_teacher: bool = True,
) -> list[RunData]:
    """
    Walk all known artifact locations and return a RunData per training run found.
    """
    runs: list[RunData] = []

    # 1) Main RL runs (TB)
    if tb_dir.exists():
        for run_name, scalars in load_all_tb_runs(tb_dir):
            rd = RunData(name=run_name, source_dir=tb_dir / run_name, run_type="rl", scalars=scalars)
            rd.config = load_config_for_run(run_name)
            runs.append(rd)

    # 2) Main RL eval
    eval_npz = DEFAULT_EVAL_DIR / "evaluations.npz"
    if eval_npz.exists() and runs:
        runs[0].eval_log = load_eval_npz(eval_npz)

    # 3) Prosthetic RL
    if include_prosthetic and PROSTHETIC_RL_DIR.exists():
        tb_sub = PROSTHETIC_RL_DIR / "tb_logs"
        scalars = load_tb_dir(tb_sub) if tb_sub.exists() else {}
        rd = RunData(name="prosthetic_rl", source_dir=PROSTHETIC_RL_DIR, run_type="prosthetic_rl", scalars=scalars)
        ev = load_eval_npz(PROSTHETIC_RL_DIR / "evaluations.npz")
        if ev is not None:
            rd.eval_log = ev
        runs.append(rd)

    # 4) Teacher
    if include_teacher and TEACHER_DIR.exists():
        tb_sub = TEACHER_DIR / "tb_logs"
        scalars = load_tb_dir(tb_sub) if tb_sub.exists() else {}
        rd = RunData(name="teacher", source_dir=TEACHER_DIR, run_type="teacher", scalars=scalars)
        ev = load_eval_npz(TEACHER_DIR / "evaluations.npz")
        if ev is not None:
            rd.eval_log = ev
        runs.append(rd)

    # 5) Supervised
    if SUPERVISED_DIR.exists():
        info = load_supervised_loss_log(SUPERVISED_DIR)
        rd = RunData(name="supervised", source_dir=SUPERVISED_DIR, run_type="supervised")
        if info:
            rd.config = info
        runs.append(rd)

    # 6) Imitation learning
    if IMITATION_DIR.exists():
        info = load_supervised_loss_log(IMITATION_DIR)
        rd = RunData(name="il", source_dir=IMITATION_DIR, run_type="il")
        if info:
            rd.config = info
        runs.append(rd)

    # 7) Scan for Monitor CSVs
    for monitor_csv in REPO_ROOT.rglob("monitor.csv"):
        parent_name = monitor_csv.parent.name
        for r in runs:
            if parent_name in r.name or r.source_dir == monitor_csv.parent:
                r.monitor_episodes = load_monitor_csv(monitor_csv)
                break

    return runs


def load_run(run_dir: Path, run_type: str = "rl") -> RunData:
    """Load a single run from an explicit directory."""
    scalars = {}
    for tb_sub in [run_dir / "tb_logs", run_dir]:
        if tb_sub.exists():
            scalars = load_tb_dir(tb_sub)
            if scalars:
                break
    rd = RunData(name=run_dir.name, source_dir=run_dir, run_type=run_type, scalars=scalars)
    for npz_name in ["evaluations.npz", "eval_logs/evaluations.npz"]:
        ev = load_eval_npz(run_dir / npz_name)
        if ev is not None:
            rd.eval_log = ev
            break
    return rd
