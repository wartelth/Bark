"""
Post-mortem analysis of trained policy networks.

Cracks open SB3 .zip checkpoints and PyTorch .pt models to extract:
  - Weight distribution stats (per layer)
  - Dead neuron detection (ReLU outputs always zero)
  - Action distribution from deterministic rollouts
  - Policy vs value network comparison
  - Gradient magnitude estimates (from weight scale)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from postpro import REPO_ROOT


@dataclass
class LayerStats:
    name: str
    shape: tuple
    n_params: int
    mean: float
    std: float
    abs_mean: float
    max_abs: float
    sparsity: float  # fraction of weights < 1e-6
    has_nan: bool


@dataclass
class PolicyAnalysis:
    model_path: str
    model_type: str  # "sb3_ppo", "sb3_sac", "pytorch"
    total_params: int = 0
    layers: list[LayerStats] = field(default_factory=list)
    dead_neurons: dict[str, int] = field(default_factory=dict)
    action_stats: Optional[dict] = None
    value_head_stats: Optional[dict] = None


def _analyze_tensor(name: str, tensor) -> LayerStats:
    arr = tensor.detach().cpu().numpy().flatten()
    return LayerStats(
        name=name,
        shape=tuple(tensor.shape),
        n_params=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        abs_mean=float(np.abs(arr).mean()),
        max_abs=float(np.abs(arr).max()),
        sparsity=float(np.mean(np.abs(arr) < 1e-6)),
        has_nan=bool(np.any(np.isnan(arr))),
    )


def _detect_dead_neurons_sb3(model) -> dict[str, int]:
    """
    For ReLU networks: a neuron is 'dead' if its bias is very negative
    and all incoming weights are small, making it unlikely to ever fire.
    Heuristic: bias < -2*std(weights) of that layer.
    """
    import torch

    dead = {}
    policy_net = model.policy
    state_dict = policy_net.state_dict()

    weight_keys = [k for k in state_dict if "weight" in k]
    for wk in weight_keys:
        bk = wk.replace("weight", "bias")
        if bk not in state_dict:
            continue
        W = state_dict[wk]
        b = state_dict[bk]
        w_std = W.std().item()
        threshold = -2.0 * w_std
        n_dead = int((b < threshold).sum().item())
        if n_dead > 0:
            dead[wk.replace(".weight", "")] = n_dead
    return dead


def analyze_sb3_model(model_path: Path) -> PolicyAnalysis:
    import torch
    from stable_baselines3 import PPO, SAC

    path_str = str(model_path)
    pa = PolicyAnalysis(model_path=path_str, model_type="sb3")

    try:
        model = PPO.load(path_str, device="cpu")
        pa.model_type = "sb3_ppo"
    except Exception:
        try:
            model = SAC.load(path_str, device="cpu")
            pa.model_type = "sb3_sac"
        except Exception as e:
            print(f"  Could not load {path_str}: {e}")
            return pa

    state_dict = model.policy.state_dict()
    total = 0
    policy_layers = []
    value_layers = []

    for name, param in state_dict.items():
        ls = _analyze_tensor(name, param)
        pa.layers.append(ls)
        total += ls.n_params
        if "value" in name or "vf" in name:
            value_layers.append(ls)
        else:
            policy_layers.append(ls)

    pa.total_params = total
    pa.dead_neurons = _detect_dead_neurons_sb3(model)

    if value_layers:
        pa.value_head_stats = {
            "n_layers": len(value_layers),
            "total_params": sum(l.n_params for l in value_layers),
            "mean_abs_weight": float(np.mean([l.abs_mean for l in value_layers])),
            "max_sparsity": float(max(l.sparsity for l in value_layers)),
        }

    return pa


def analyze_pytorch_model(model_path: Path, model_class=None) -> PolicyAnalysis:
    import torch

    pa = PolicyAnalysis(model_path=str(model_path), model_type="pytorch")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    total = 0
    for name, param in state_dict.items():
        t = param if isinstance(param, torch.Tensor) else torch.tensor(param)
        ls = _analyze_tensor(name, t)
        pa.layers.append(ls)
        total += ls.n_params
    pa.total_params = total

    weight_keys = [k for k in state_dict if "weight" in k]
    for wk in weight_keys:
        bk = wk.replace("weight", "bias")
        if bk not in state_dict:
            continue
        W = state_dict[wk]
        b = state_dict[bk]
        w_std = W.std().item()
        threshold = -2.0 * w_std
        n_dead = int((b < threshold).sum().item())
        if n_dead > 0:
            pa.dead_neurons[wk.replace(".weight", "")] = n_dead

    return pa


# ---------------------------------------------------------------------------
# Action distribution from rollouts
# ---------------------------------------------------------------------------

def rollout_action_stats(
    model_path: Path,
    env_id: str = "BarkAnt3Leg-v0",
    n_episodes: int = 10,
    model_type: str = "sb3",
) -> dict:
    """Roll out a policy and collect action statistics."""
    import gymnasium as gym

    if model_type == "sb3":
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
    else:
        return {}

    try:
        import envs
        envs.register_bark_envs()
    except Exception:
        pass

    try:
        env = gym.make(env_id)
    except Exception:
        return {}

    all_actions = []
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action.copy())
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            if term or trunc:
                break
        rewards.append(ep_reward)

    env.close()
    actions = np.array(all_actions)

    stats = {
        "n_steps": len(actions),
        "n_episodes": n_episodes,
        "action_dim": actions.shape[1] if actions.ndim > 1 else 1,
        "mean_reward": float(np.mean(rewards)),
        "per_dim_mean": actions.mean(axis=0).tolist(),
        "per_dim_std": actions.std(axis=0).tolist(),
        "per_dim_min": actions.min(axis=0).tolist(),
        "per_dim_max": actions.max(axis=0).tolist(),
        "action_magnitude_mean": float(np.abs(actions).mean()),
        "action_correlation": np.corrcoef(actions.T).tolist() if actions.ndim > 1 else None,
    }
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_weight_distributions(
    analyses: list[PolicyAnalysis],
    out_dir: Path,
    filename: str = "weight_distributions.png",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not analyses:
        return

    fig, axes = plt.subplots(len(analyses), 1, figsize=(14, 4 * len(analyses)), squeeze=False)

    for row, pa in enumerate(analyses):
        ax = axes[row][0]
        weight_layers = [l for l in pa.layers if "weight" in l.name and l.n_params > 1]
        if not weight_layers:
            continue

        positions = range(len(weight_layers))
        means = [l.mean for l in weight_layers]
        stds = [l.std for l in weight_layers]
        names = [l.name.split(".")[-2] if "." in l.name else l.name for l in weight_layers]

        ax.bar(positions, means, yerr=stds, capsize=3, color="#2196F3", alpha=0.7, label="mean +/- std")

        for i, l in enumerate(weight_layers):
            if l.has_nan:
                ax.annotate("NaN!", (i, means[i]), fontsize=8, color="red", ha="center")
            if l.sparsity > 0.5:
                ax.annotate(f"sparse:{l.sparsity:.0%}", (i, means[i]),
                            fontsize=7, color="orange", ha="center", va="bottom")

        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{Path(pa.model_path).name} ({pa.total_params:,} params, {pa.model_type})")
        ax.set_ylabel("Weight Value")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)

        if pa.dead_neurons:
            dead_text = ", ".join(f"{k}: {v}" for k, v in pa.dead_neurons.items())
            ax.annotate(f"Dead neurons: {dead_text}", xy=(0.01, 0.97), xycoords="axes fraction",
                        fontsize=7, color="red", va="top")

    fig.suptitle("Policy Weight Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_action_distribution(
    action_stats: dict,
    label: str,
    out_dir: Path,
    filename: str = "action_distribution.png",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not action_stats or "per_dim_mean" not in action_stats:
        return

    n_dims = action_stats["action_dim"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n_dims)
    means = action_stats["per_dim_mean"]
    stds = action_stats["per_dim_std"]
    mins = action_stats["per_dim_min"]
    maxs = action_stats["per_dim_max"]

    axes[0].bar(x, means, yerr=stds, capsize=3, color="#4CAF50", alpha=0.7)
    axes[0].scatter(x, mins, color="blue", s=15, zorder=5, label="min")
    axes[0].scatter(x, maxs, color="red", s=15, zorder=5, label="max")
    axes[0].set_xlabel("Action Dimension")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{label}: Action Stats per Dimension")
    axes[0].legend(fontsize=8)

    corr = action_stats.get("action_correlation")
    if corr is not None:
        im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[1].set_title("Action Correlation Matrix")
        axes[1].set_xlabel("Action Dim")
        axes[1].set_ylabel("Action Dim")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


# ---------------------------------------------------------------------------
# Master entry
# ---------------------------------------------------------------------------

def analyze_all_policies(out_dir: Path, run_rollouts: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Policy Analysis ---\n")

    analyses = []
    model_locations = [
        (REPO_ROOT / "models" / "best" / "best_model.zip", "rl_best"),
        (REPO_ROOT / "models" / "final_model.zip", "rl_final"),
        (REPO_ROOT / "models" / "prosthetic_rl" / "best_model.zip", "prosthetic_rl"),
        (REPO_ROOT / "pretrained" / "go1_teacher" / "best_model.zip", "teacher"),
    ]

    for path, label in model_locations:
        if not path.exists():
            continue
        print(f"  Analyzing {label}: {path}")
        pa = analyze_sb3_model(path)
        if pa.total_params > 0:
            analyses.append(pa)
            print(f"    {pa.total_params:,} params, {len(pa.dead_neurons)} layers with dead neurons")

    pt_path = REPO_ROOT / "models" / "supervised_prosthetic" / "best_model.pt"
    if pt_path.exists():
        print(f"  Analyzing supervised: {pt_path}")
        pa = analyze_pytorch_model(pt_path)
        if pa.total_params > 0:
            analyses.append(pa)
            print(f"    {pa.total_params:,} params")

    if analyses:
        plot_weight_distributions(analyses, out_dir)

    if run_rollouts:
        for path, label in model_locations[:2]:
            if not path.exists():
                continue
            print(f"  Rolling out {label} for action stats...")
            stats = rollout_action_stats(path, n_episodes=5)
            if stats:
                plot_action_distribution(stats, label, out_dir, f"actions_{label}.png")
                print(f"    Mean reward: {stats['mean_reward']:.2f}, "
                      f"action magnitude: {stats['action_magnitude_mean']:.4f}")

    return analyses


def print_policy_summary(analyses: list[PolicyAnalysis]):
    for pa in analyses:
        print(f"\n{'='*60}")
        print(f"  {Path(pa.model_path).name} ({pa.model_type})")
        print(f"  Total parameters: {pa.total_params:,}")
        print(f"{'='*60}")
        print(f"  {'Layer':<40} {'Shape':<20} {'AbsMean':>8} {'Std':>8} {'Sparse%':>8}")
        print(f"  {'-'*84}")
        for l in pa.layers:
            if l.n_params < 2:
                continue
            print(f"  {l.name:<40} {str(l.shape):<20} {l.abs_mean:>8.5f} {l.std:>8.5f} {l.sparsity*100:>7.1f}%")
        if pa.dead_neurons:
            print(f"\n  Dead neurons:")
            for layer, count in pa.dead_neurons.items():
                print(f"    {layer}: {count} dead")
        if pa.value_head_stats:
            v = pa.value_head_stats
            print(f"\n  Value head: {v['total_params']:,} params, "
                  f"mean |w|={v['mean_abs_weight']:.5f}, max sparsity={v['max_sparsity']:.1%}")
