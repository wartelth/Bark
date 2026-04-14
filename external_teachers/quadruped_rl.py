from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from stable_baselines3 import PPO

QUADRUPED_RL_ROOT = Path(r"D:\quadruped-rl-locomotion")
DEFAULT_WORKING_MODEL = QUADRUPED_RL_ROOT / "models" / "2024-04-27_18-04-12=1_pos_ctrl_20mil_iter_walking_with_fast_steps" / "best_model.zip"
ALTERNATE_WORKING_MODELS = [
    QUADRUPED_RL_ROOT / "models" / "2024-04-28_22-10-59=1_pos_ctrl_20mil_iter_walking_with_normal_steps" / "best_model.zip",
    QUADRUPED_RL_ROOT / "models" / "2024-04-21_20-04-20=1_pso_ctrl_30mil_iter_walking_with_oriented_rear_legs" / "best_model.zip",
]


@contextmanager
def _repo_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _ensure_imports():
    if str(QUADRUPED_RL_ROOT) not in sys.path:
        sys.path.insert(0, str(QUADRUPED_RL_ROOT))


def make_env(ctrl_type: str = "position", render_mode: str | None = None):
    _ensure_imports()
    with _repo_cwd(QUADRUPED_RL_ROOT):
        from go1_mujoco_env import Go1MujocoEnv
        return Go1MujocoEnv(ctrl_type=ctrl_type, render_mode=render_mode)


def load_legacy_ppo(model_path: str | Path, env=None):
    """
    Load older PPO checkpoints that serialize schedule callables incompatibly on newer
    Python/SB3 versions. These fields are only needed for training, not inference.
    """
    custom_objects = {
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0,
    }
    return PPO.load(str(model_path), env=env, device="cpu", custom_objects=custom_objects)


def load_teacher(model_path: str | Path | None = None, ctrl_type: str = "position"):
    _ensure_imports()
    mp = Path(model_path) if model_path else DEFAULT_WORKING_MODEL
    env = make_env(ctrl_type=ctrl_type, render_mode=None)
    model = load_legacy_ppo(mp, env=env)
    env.close()
    return model


def benchmark_model(model_path: str | Path | None = None, ctrl_type: str = "position", episodes: int = 5):
    import numpy as np

    mp = Path(model_path) if model_path else DEFAULT_WORKING_MODEL
    env = make_env(ctrl_type=ctrl_type, render_mode=None)
    model = load_legacy_ppo(mp, env=env)
    lens, rets = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ret = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            ret += float(reward)
            steps += 1
            if term or trunc:
                break
        lens.append(steps)
        rets.append(ret)
    env.close()
    return {
        "model_path": str(mp),
        "episodes": episodes,
        "mean_length": float(np.mean(lens)),
        "mean_return": float(np.mean(rets)),
        "lengths": lens,
        "returns": rets,
    }
