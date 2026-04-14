"""
Side-by-side visualization: teacher (all 4 legs) vs hybrid (teacher 0-2 + student leg 3).

Usage:
    PYTHONPATH=. python evaluate/visualize_hybrid.py
    PYTHONPATH=. python evaluate/visualize_hybrid.py --student supervised
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrained.load_teacher import (
    load_teacher, make_go1_env, split_obs_and_action,
    GO1_LEG3_JOINT_IX,
)
from train.train_supervised import ProstheticMLP

REPO = Path(__file__).resolve().parent.parent


def run_teacher(teacher, env, steps: int):
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action, _ = teacher.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()


def run_hybrid_supervised(teacher, student_model, env, steps: int):
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, _ = split_obs_and_action(obs, teacher_action)

        with torch.no_grad():
            t = torch.from_numpy(obs_3leg).unsqueeze(0)
            student_leg3 = student_model(t).squeeze(0).numpy()

        combined = teacher_action.copy()
        combined[GO1_LEG3_JOINT_IX] = student_leg3

        obs, _, term, trunc, _ = env.step(combined)
        env.render()
        if term or trunc:
            obs, _ = env.reset()


def run_hybrid_rl(teacher, rl_model, env, steps: int):
    from stable_baselines3 import PPO
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, _ = split_obs_and_action(obs, teacher_action)

        student_leg3, _ = rl_model.predict(obs_3leg, deterministic=True)

        combined = teacher_action.copy()
        combined[GO1_LEG3_JOINT_IX] = student_leg3

        obs, _, term, trunc, _ = env.step(combined)
        env.render()
        if term or trunc:
            obs, _ = env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=["supervised", "rl", "teacher"], default="teacher")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--teacher-path", type=str, default=None)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher_path)
    env = make_go1_env(render=True)

    if args.student == "teacher":
        print("Running teacher only (all 4 legs)...")
        run_teacher(teacher, env, args.steps)

    elif args.student == "supervised":
        sup_dir = REPO / "models" / "supervised_prosthetic"
        obs, _ = env.reset()
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, _ = split_obs_and_action(obs, teacher_action)
        obs_dim = obs_3leg.shape[0]

        student = ProstheticMLP(obs_dim, action_dim=3)
        student.load_state_dict(
            torch.load(sup_dir / "best_model.pt", map_location="cpu", weights_only=True)
        )
        student.eval()
        print("Running hybrid: teacher legs 0-2 + supervised student leg 3...")
        run_hybrid_supervised(teacher, student, env, args.steps)

    elif args.student == "rl":
        from stable_baselines3 import PPO
        rl_path = REPO / "models" / "prosthetic_rl" / "prosthetic_rl_final.zip"
        rl_model = PPO.load(str(rl_path))
        print("Running hybrid: teacher legs 0-2 + RL student leg 3...")
        run_hybrid_rl(teacher, rl_model, env, args.steps)

    env.close()


if __name__ == "__main__":
    main()
