"""
Canonical teacher interface for Bark.

This now points to the proven walking Go1 policy from the local
`D:\\quadruped-rl-locomotion` repo and exposes a consistent API for:
  - teacher loading
  - Go1 env creation
  - 3-leg observation masking

The only experiment we care about is whether supervised learning or RL
best follows the teacher, so this module optimizes for that workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from external_teachers.quadruped_rl import DEFAULT_WORKING_MODEL, load_teacher as load_external_teacher, make_env

DEFAULT_MODEL = DEFAULT_WORKING_MODEL

GO1_LEG3_JOINT_IX = [9, 10, 11]
N_JOINTS = 12

# External Go1 env observation layout:
# [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), desired_vel(3),
#  dofs_position(12), dofs_velocity(12), last_action(12)]
OBS_DOF_POS_START = 12
OBS_DOF_VEL_START = 24
OBS_LAST_ACTION_START = 36

LEG3_OBS_REMOVE_IX = (
    [OBS_DOF_POS_START + j for j in GO1_LEG3_JOINT_IX]
    + [OBS_DOF_VEL_START + j for j in GO1_LEG3_JOINT_IX]
    + [OBS_LAST_ACTION_START + j for j in GO1_LEG3_JOINT_IX]
)


def load_teacher(model_path: str | Path | None = None):
    """Load the working external Go1 teacher by default."""
    return load_external_teacher(model_path, ctrl_type="position")


def make_go1_env(render: bool = False, render_mode: str | None = None):
    """Create the external Go1 env used by the working teacher."""
    if render_mode is None:
        render_mode = "human" if render else None
    return make_env(ctrl_type="position", render_mode=render_mode)


def split_obs_and_action(obs: np.ndarray, action: np.ndarray):
    """Split full obs/action into 3-leg input and leg-3 target for the external Go1 env."""
    obs_3leg = np.delete(obs, LEG3_OBS_REMOVE_IX).astype(np.float32)
    action_leg3 = action[GO1_LEG3_JOINT_IX].astype(np.float32)
    return obs_3leg, action_leg3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    teacher = load_teacher(args.model)
    env = make_go1_env(render=args.render)

    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for step in range(args.steps):
        action, _ = teacher.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            print(f"Episode ended at step {step}, reward={total_reward:.1f}")
            obs, _ = env.reset(seed=step + 1)
            total_reward = 0.0

    env.close()
    print(f"Finished {args.steps} steps.")


if __name__ == "__main__":
    main()
