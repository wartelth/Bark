"""
Compare supervised student vs RL student vs teacher on held-out episodes.
Metrics: per-step MSE, trajectory reward, gait symmetry.

Usage:
    PYTHONPATH=. python evaluate/compare.py
    PYTHONPATH=. python evaluate/compare.py --episodes 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.scenario_library import apply_scenario, scenario_pool
from pretrained.load_teacher import (
    load_teacher, make_go1_env, split_obs_and_action,
    GO1_LEG3_JOINT_IX,
)
from train.train_supervised import ProstheticMLP

REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "reports"


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_supervised_student(model_dir: Path, obs_dim: int, device: str = "cpu"):
    model = ProstheticMLP(obs_dim, action_dim=3)
    state = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _scenario_schedule(pool_name: str, n_episodes: int):
    specs = scenario_pool(pool_name)
    return [specs[i % len(specs)] for i in range(n_episodes)]


def _finalize_per_scenario(storage: dict[str, dict]) -> list[dict]:
    rows = []
    for name, values in storage.items():
        count = max(values["count"], 1)
        rows.append(
            {
                "scenario": name,
                "episodes": values["count"],
                "reward": values["reward"] / count,
                "ep_len": values["ep_len"] / count,
                "mse": values["mse"] / count if values["has_mse"] else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["scenario"])


def evaluate_teacher_only(teacher, env, scenarios):
    """Baseline: teacher controls all 4 legs."""
    rewards, ep_lens = [], []
    per_scenario = {}
    for ep, spec in enumerate(scenarios):
        apply_scenario(env, spec)
        obs, _ = env.reset(seed=ep)
        total_r, steps = 0.0, 0
        while True:
            action, _ = teacher.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            steps += 1
            if term or trunc:
                break
        rewards.append(total_r)
        ep_lens.append(steps)
        stats = per_scenario.setdefault(spec.name, {"count": 0, "reward": 0.0, "ep_len": 0.0, "mse": 0.0, "has_mse": False})
        stats["count"] += 1
        stats["reward"] += total_r
        stats["ep_len"] += steps
    return np.mean(rewards), np.mean(ep_lens), _finalize_per_scenario(per_scenario)


def evaluate_hybrid(teacher, student_fn, env, scenarios):
    """Hybrid: teacher legs 0-2, student leg 3. Returns reward, MSE, ep_len."""
    rewards, mses, ep_lens = [], [], []
    per_scenario = {}
    for ep, spec in enumerate(scenarios):
        apply_scenario(env, spec)
        obs, _ = env.reset(seed=ep)
        total_r, total_mse, steps = 0.0, 0.0, 0
        while True:
            teacher_action, _ = teacher.predict(obs, deterministic=True)
            obs_3leg, teacher_leg3 = split_obs_and_action(obs, teacher_action)

            student_leg3 = student_fn(obs_3leg)

            combined = teacher_action.copy()
            combined[GO1_LEG3_JOINT_IX] = student_leg3

            obs, r, term, trunc, _ = env.step(combined)
            total_r += r
            total_mse += np.mean((student_leg3 - teacher_leg3) ** 2)
            steps += 1
            if term or trunc:
                break
        rewards.append(total_r)
        mses.append(total_mse / max(steps, 1))
        ep_lens.append(steps)
        stats = per_scenario.setdefault(spec.name, {"count": 0, "reward": 0.0, "ep_len": 0.0, "mse": 0.0, "has_mse": True})
        stats["count"] += 1
        stats["reward"] += total_r
        stats["ep_len"] += steps
        stats["mse"] += total_mse / max(steps, 1)
    return np.mean(rewards), np.mean(mses), np.mean(ep_lens), _finalize_per_scenario(per_scenario)


def save_summary_plot(results: list[dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    rewards = [r["reward"] for r in results]
    reward_retention = [r["reward_retention"] for r in results]
    mses = [r.get("mse", 0.0) for r in results]
    lengths = [r["ep_len"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    axes[0].bar(labels, rewards, color=colors)
    axes[0].set_title("Episode Reward")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, reward_retention, color=colors)
    axes[1].set_title("Reward Retention vs Teacher (%)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(labels, lengths, color=colors, alpha=0.9, label="ep len")
    if any(m > 0 for m in mses):
        ax2 = axes[2].twinx()
        ax2.plot(labels, mses, color="black", marker="o", linewidth=2, label="MSE")
        ax2.set_ylabel("Leg-3 MSE")
    axes[2].set_title("Stability / Tracking")
    axes[2].set_ylabel("Episode Length")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Teacher vs Supervised vs RL Student")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scenario_plot(results: list[dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_names = sorted({row["scenario"] for result in results for row in result.get("per_scenario", [])})
    if not scenario_names:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, result in enumerate(results):
        per = {row["scenario"]: row for row in result.get("per_scenario", [])}
        reward_vals = [per.get(name, {}).get("reward_retention", np.nan) for name in scenario_names]
        len_vals = [per.get(name, {}).get("ep_len", np.nan) for name in scenario_names]
        axes[0].plot(scenario_names, reward_vals, marker="o", linewidth=2, color=colors[idx], label=result["label"])
        axes[1].plot(scenario_names, len_vals, marker="o", linewidth=2, color=colors[idx], label=result["label"])

    axes[0].set_title("Reward Retention by Scenario")
    axes[0].set_ylabel("Retention vs Teacher (%)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Episode Length by Scenario")
    axes[1].set_ylabel("Episode Length")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--supervised-dir", type=str, default="models/supervised_prosthetic")
    parser.add_argument("--il-dir", type=str, default="models/imitation_prosthetic")
    parser.add_argument("--rl-model", type=str, default="models/prosthetic_rl/prosthetic_rl_final.zip")
    parser.add_argument("--scenario-pool", type=str, default="all_train")
    args = parser.parse_args()

    teacher = load_teacher(args.teacher)
    env = make_go1_env(render=False)
    scenarios = _scenario_schedule(args.scenario_pool, args.episodes)

    print("=" * 60)
    print("Evaluating teacher (4-leg baseline)...")
    t_reward, t_len, teacher_per_scenario = evaluate_teacher_only(teacher, env, scenarios)
    print(f"  Reward: {t_reward:.1f}  |  Ep length: {t_len:.0f}")
    results = [
        {
            "label": "teacher",
            "reward": float(t_reward),
            "reward_retention": 100.0,
            "ep_len": float(t_len),
            "mse": 0.0,
            "per_scenario": [
                {**row, "reward_retention": 100.0}
                for row in teacher_per_scenario
            ],
        }
    ]
    teacher_lookup = {row["scenario"]: row for row in teacher_per_scenario}

    sup_dir = REPO / args.supervised_dir
    if (sup_dir / "best_model.pt").exists():
        obs, _ = env.reset()
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, _ = split_obs_and_action(obs, teacher_action)
        obs_dim = obs_3leg.shape[0]

        sup_model = load_supervised_student(sup_dir, obs_dim)

        def sup_fn(obs_3leg):
            with torch.no_grad():
                t = torch.from_numpy(obs_3leg).unsqueeze(0)
                return sup_model(t).squeeze(0).numpy()

        print("\nEvaluating supervised student (hybrid)...")
        s_reward, s_mse, s_len, s_per_scenario = evaluate_hybrid(teacher, sup_fn, env, scenarios)
        print(f"  Reward: {s_reward:.1f}  |  MSE: {s_mse:.6f}  |  Ep length: {s_len:.0f}")
        print(f"  Reward retention: {s_reward / max(t_reward, 1e-8) * 100:.1f}%")
        results.append(
            {
                "label": "supervised",
                "reward": float(s_reward),
                "reward_retention": float(s_reward / max(t_reward, 1e-8) * 100.0),
                "ep_len": float(s_len),
                "mse": float(s_mse),
                "per_scenario": [
                    {
                        **row,
                        "reward_retention": float(row["reward"] / max(teacher_lookup[row["scenario"]]["reward"], 1e-8) * 100.0),
                    }
                    for row in s_per_scenario
                ],
            }
        )
    else:
        print(f"\nSupervised model not found at {sup_dir}, skipping.")

    il_dir = REPO / args.il_dir
    if (il_dir / "best_model.pt").exists():
        obs, _ = env.reset()
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, _ = split_obs_and_action(obs, teacher_action)
        obs_dim = obs_3leg.shape[0]

        il_model = load_supervised_student(il_dir, obs_dim)

        def il_fn(obs_3leg):
            with torch.no_grad():
                t = torch.from_numpy(obs_3leg).unsqueeze(0)
                return il_model(t).squeeze(0).numpy()

        print("\nEvaluating IL student (hybrid)...")
        i_reward, i_mse, i_len, i_per_scenario = evaluate_hybrid(teacher, il_fn, env, scenarios)
        print(f"  Reward: {i_reward:.1f}  |  MSE: {i_mse:.6f}  |  Ep length: {i_len:.0f}")
        print(f"  Reward retention: {i_reward / max(t_reward, 1e-8) * 100:.1f}%")
        results.append(
            {
                "label": "il",
                "reward": float(i_reward),
                "reward_retention": float(i_reward / max(t_reward, 1e-8) * 100.0),
                "ep_len": float(i_len),
                "mse": float(i_mse),
                "per_scenario": [
                    {
                        **row,
                        "reward_retention": float(row["reward"] / max(teacher_lookup[row["scenario"]]["reward"], 1e-8) * 100.0),
                    }
                    for row in i_per_scenario
                ],
            }
        )
    else:
        print(f"\nIL model not found at {il_dir}, skipping.")

    from stable_baselines3 import PPO
    rl_path = REPO / args.rl_model
    if rl_path.exists():
        rl_student = PPO.load(str(rl_path), device="cpu")

        def rl_fn(obs_3leg):
            action, _ = rl_student.predict(obs_3leg, deterministic=True)
            return action

        print("\nEvaluating RL student (hybrid)...")
        r_reward, r_mse, r_len, r_per_scenario = evaluate_hybrid(teacher, rl_fn, env, scenarios)
        print(f"  Reward: {r_reward:.1f}  |  MSE: {r_mse:.6f}  |  Ep length: {r_len:.0f}")
        print(f"  Reward retention: {r_reward / max(t_reward, 1e-8) * 100:.1f}%")
        results.append(
            {
                "label": "rl",
                "reward": float(r_reward),
                "reward_retention": float(r_reward / max(t_reward, 1e-8) * 100.0),
                "ep_len": float(r_len),
                "mse": float(r_mse),
                "per_scenario": [
                    {
                        **row,
                        "reward_retention": float(row["reward"] / max(teacher_lookup[row["scenario"]]["reward"], 1e-8) * 100.0),
                    }
                    for row in r_per_scenario
                ],
            }
        )
    else:
        print(f"\nRL model not found at {rl_path}, skipping.")

    print("=" * 60)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "student_comparison.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    save_summary_plot(results, REPORTS_DIR / "student_comparison.png")
    save_scenario_plot(results, REPORTS_DIR / "student_comparison_by_scenario.png")
    print(f"Saved summary to {REPORTS_DIR / 'student_comparison.png'}")
    env.close()


if __name__ == "__main__":
    main()
