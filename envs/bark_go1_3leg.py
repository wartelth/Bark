"""
BarkGo1_3Leg: Unitree Go1 dog-like quadruped with observation restricted to 3 legs.
The policy sees only legs 0 (FR), 1 (FL), 2 (RR) and must output full 12D action (including leg 3 RL),
so it learns how the 4th leg depends on the other three — same prosthetic setup as BarkAnt3Leg but with a
realistic dog-like robot (Go1).

Requires the MuJoCo Menagerie Go1 model. See README for setup (clone mujoco_menagerie into third_party/).
"""
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv

# Go1: 12 joints = 4 legs × 3 (hip, thigh, calf). Order: FR, FL, RR, RL.
# Leg 3 = RL = last 3 joints (indices 9, 10, 11).
N_JOINTS_GO1 = 12
PROSTHETIC_LEG_JOINT_IX = [9, 10, 11]  # RL leg


def _go1_leg3_obs_indices(
    nq: int,
    nv: int,
    exclude_current_positions: bool,
) -> list[int]:
    """
    Compute observation indices to remove for leg 3 (RL).
    Ant-v5 obs = position (qpos[2:] if exclude) + velocity (qvel) + optional cfrc_ext.
    Go1: 7 root (x,y,z, quat) + 12 joints → nq=19, nv=18.
    """
    skip = 2 if exclude_current_positions else 0
    n_pos = nq - skip  # 17
    joint_pos_start = 5  # 0=z, 1-4=quat, 5-16=12 joints
    joint_vel_start = n_pos + 6  # 6 = torso linear + angular vel
    remove_ix = []
    for j in PROSTHETIC_LEG_JOINT_IX:
        remove_ix.append(joint_pos_start + j)
    for j in PROSTHETIC_LEG_JOINT_IX:
        remove_ix.append(joint_vel_start + j)
    return remove_ix


def _mask_go1_obs_to_3_legs(obs: np.ndarray, remove_ix: list[int]) -> np.ndarray:
    return np.delete(obs, remove_ix).astype(np.float32)


def _default_go1_xml_path() -> Path | None:
    """Return path to Go1 scene.xml if menagerie is present."""
    repo = Path(__file__).resolve().parent.parent
    candidates = [
        repo / "third_party" / "mujoco_menagerie" / "unitree_go1" / "scene.xml",
        repo / "assets" / "unitree_go1" / "scene.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class BarkGo1_3LegEnv(AntEnv):
    """
    Unitree Go1 quadruped (dog-like) with observation masked to 3 legs (leg 3 / RL hidden).
    Action space 12D: policy must infer leg 3 action from the other three legs.
    """

    def __init__(
        self,
        xml_file: str | Path | None = None,
        prosthetic_leg_index: int = 3,
        obs_noise_std: float = 0.0,
        **kwargs,
    ):
        self._prosthetic_leg_index = prosthetic_leg_index
        self._obs_noise_std = obs_noise_std

        if xml_file is None:
            default = _default_go1_xml_path()
            if default is None:
                raise FileNotFoundError(
                    "Go1 model not found. Clone MuJoCo Menagerie into third_party:\n"
                    "  git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie\n"
                    "Or place unitree_go1 (scene.xml + go1.xml) in assets/unitree_go1/."
                )
            xml_file = str(default)
        else:
            xml_file = str(Path(xml_file).resolve())

        # Go1-specific defaults (from Gymnasium tutorial)
        kwargs.setdefault("frame_skip", 25)  # dt ≈ 0.05s
        kwargs.setdefault("healthy_z_range", (0.195, 0.75))
        kwargs.setdefault("reset_noise_scale", 0.1)
        kwargs.setdefault("ctrl_cost_weight", 0.05)
        kwargs.setdefault("exclude_current_positions_from_observation", False)
        kwargs.setdefault("include_cfrc_ext_in_observation", False)

        super().__init__(xml_file=xml_file, **kwargs)

        nq = self.model.nq
        nv = self.model.nv
        self._leg3_remove_ix = _go1_leg3_obs_indices(
            nq,
            nv,
            self._exclude_current_positions_from_observation,
        )
        parent_obs_size = int(self.observation_space.shape[0])
        new_obs_size = parent_obs_size - len(self._leg3_remove_ix)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_obs_size,),
            dtype=np.float32,
        )

    def _get_obs(self):
        obs = super()._get_obs()
        obs = _mask_go1_obs_to_3_legs(obs, self._leg3_remove_ix)
        if self._obs_noise_std > 0 and self.np_random is not None:
            obs = obs + self.np_random.normal(
                0, self._obs_noise_std, obs.shape
            ).astype(np.float32)
        return obs

    def step(self, action):
        return super().step(action)


def register_bark_go1_envs():
    from gymnasium.envs.registration import register

    register(
        id="BarkGo1_3Leg-v0",
        entry_point="envs.bark_go1_3leg:BarkGo1_3LegEnv",
        max_episode_steps=1000,
        kwargs={},
    )
