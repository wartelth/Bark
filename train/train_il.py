"""
Imitation learning: BC or GAIL on BarkAnt3Leg using expert demos.
Expert data can be from rollouts (e.g. a trained policy or scripted policy) or from
jacket-derived reference (after mapping to env state space).
Usage:
  PYTHONPATH=. python -m train.train_il --config configs/bc_ant_3leg.yaml --expert_path demos/expert_rollouts.npz
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from envs import register_bark_envs

register_bark_envs()

import gymnasium as gym


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_expert_rollouts(
    env_id: str,
    n_trajectories: int = 50,
    max_steps: int = 500,
    seed: int = 0,
    policy=None,
    save_path: Optional[str] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Collect (obs_list, act_list) from env. If policy is None, use random policy for demo.
    """
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    for _ in range(n_trajectories):
        obs, _ = env.reset(seed=int(rng.integers(0, 1e9)))
        o_traj, a_traj = [obs], []
        for _ in range(max_steps - 1):
            if policy is not None:
                act, _ = policy.predict(obs, deterministic=True)
            else:
                act = env.action_space.sample()
            a_traj.append(act)
            obs, _, term, trunc, _ = env.step(act)
            o_traj.append(obs)
            if term or trunc:
                break
        obs_list.append(np.array(o_traj))
        act_list.append(np.array(a_traj))
    env.close()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            obs=[o for o in obs_list],
            acts=[a for a in act_list],
        )
    return obs_list, act_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc_ant_3leg.yaml")
    parser.add_argument("--expert_path", type=str, default=None, help=".npz with 'obs' and 'acts' arrays")
    parser.add_argument("--collect_demos", type=int, default=0, help="If >0, collect this many random demos first and save to expert_path")
    parser.add_argument("--algorithm", type=str, choices=["bc", "gail"], default="bc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _repo_root / config_path
    config = load_config(str(config_path))
    env_id = config.get("env_id", "BarkAnt3Leg-v0")
    collect_demos = getattr(args, "collect_demos", 0)
    if collect_demos > 0:
        expert_path = args.expert_path or "demos/expert_rollouts.npz"
        if not Path(expert_path).is_absolute():
            expert_path = str(_repo_root / expert_path)
        collect_expert_rollouts(
            env_id,
            n_trajectories=collect_demos,
            max_steps=config.get("max_episode_steps", 500),
            seed=args.seed,
            save_path=expert_path,
        )
        print(f"Saved {collect_demos} demos to {expert_path}")
        return

    expert_path = args.expert_path
    if expert_path:
        expert_path = Path(expert_path)
        if not expert_path.is_absolute():
            # Try repo root first, then cwd (for python -m from repo root)
            for base in (_repo_root, Path.cwd()):
                p = base / expert_path
                if p.exists():
                    expert_path = p
                    break
            else:
                expert_path = _repo_root / expert_path
    else:
        expert_path = None
    if not expert_path or not expert_path.exists():
        if expert_path:
            print(f"Expert file not found: {expert_path}. Tried repo root: {_repo_root}")
        else:
            print("No expert_path provided. Run with --collect_demos 30 to create demos first.")
        return

    data = np.load(str(expert_path), allow_pickle=True)
    assert "obs" in data and "acts" in data, "npz must contain 'obs' and 'acts'"
    expert_obs = data["obs"].tolist() if data["obs"].ndim == 0 else list(data["obs"])
    expert_acts = data["acts"].tolist() if data["acts"].ndim == 0 else list(data["acts"])
    print(f"Loaded {len(expert_obs)} expert trajectories", flush=True)

    try:
        from imitation.algorithms import bc
        from imitation.data import rollout
        from imitation.data.types import TrajectoryWithRew
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        print("imitation library not installed: pip install imitation", e)
        return

    env = gym.make(env_id)
    # Build imitation trajectories (reward can be dummy)
    trajs = []
    for o, a in zip(expert_obs, expert_acts):
        if len(o) < 2 or len(a) < 1:
            continue
        # terminal: True if trajectory ended (e.g. done); imitation expects this
        trajs.append(
            TrajectoryWithRew(
                obs=o,
                acts=a,
                rews=np.zeros(len(a)),
                infos=None,
                terminal=True,
            )
        )
    if not trajs:
        print("No valid trajectories")
        return

    n_transitions = sum(len(t.acts) for t in trajs)
    batch_size = min(config.get("batch_size", 64), max(1, n_transitions // 2))
    rng = np.random.default_rng(args.seed)
    lr = config.get("learning_rate", 3e-4)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajs,
        rng=rng,
        batch_size=batch_size,
        optimizer_kwargs=dict(lr=lr),
    )
    n_epochs = config.get("n_epochs", 10)
    bc_trainer.train(n_epochs=n_epochs)
    env.close()

    save_path = Path(config.get("save_path", "models/il_bc"))
    save_path.mkdir(parents=True, exist_ok=True)
    bc_trainer.policy.save(save_path / "policy")
    print("BC policy saved to", save_path, flush=True)
