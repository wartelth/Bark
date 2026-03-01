"""
BarkAnt3Leg: Ant quadruped with observation restricted to 3 legs (and torso).
The policy sees only legs 0, 1, 2 state and must output full 8D action (including leg 3),
so it learns how the 4th leg depends on the other three.
"""
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv


# Ant default obs (exclude_current_positions): 13 pos (z, quat(4), joints(8)) + 14 vel
# pos indices: 0=z, 1-4=quat, 5-12=joints (leg0: 5,6  leg1: 7,8  leg2: 9,10  leg3: 11,12)
# vel indices: 13-14=torso_vel(2), 15-17=torso_angvel(3)? Need to re-check.
# Actually Ant qpos: 15 (x,y,z, quat(4), 8 joints). With exclude: position = position[2:] -> 13 (z, quat4, 8 joints)
# qvel: 14 (vx,vy,vz, angvel(3), 8 joint vels). So concat: [pos 13, vel 14] = 27.
POS_DIM = 13   # z(1) + quat(4) + 8 joints
VEL_DIM = 14   # 3 torso vel + 3 angvel + 8 joint vel
# Indices in obs (27): pos 0..12, vel 13..26. Leg 3 = last 2 joints in each.
LEG3_POS_IX = [11, 12]
LEG3_VEL_IX = [11, 12]  # within the 14-dim vel slice


def _mask_obs_to_3_legs(obs: np.ndarray) -> np.ndarray:
    """Remove leg 3 (prosthetic) from observation so policy sees only 3 legs + torso."""
    pos = obs[:POS_DIM]
    vel = obs[POS_DIM:]
    pos_3leg = np.delete(pos, LEG3_POS_IX)
    vel_3leg = np.delete(vel, LEG3_VEL_IX)
    return np.concatenate([pos_3leg, vel_3leg]).astype(np.float32)


class BarkAnt3LegEnv(AntEnv):
    """
    Ant environment with observation space reduced to torso + 3 legs (leg 3 hidden).
    Action space unchanged (8D): policy must infer leg 3 action from other legs.
    """

    def __init__(
        self,
        prosthetic_leg_index: int = 3,
        obs_noise_std: float = 0.0,
        **kwargs,
    ):
        self._prosthetic_leg_index = prosthetic_leg_index
        self._obs_noise_std = obs_noise_std
        super().__init__(**kwargs)
        # Observation: 27 - 2 (pos leg3) - 2 (vel leg3) = 23
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32,
        )

    def _get_obs(self):
        obs = super()._get_obs()
        obs = _mask_obs_to_3_legs(obs)
        if self._obs_noise_std > 0 and self.np_random is not None:
            obs = obs + self.np_random.normal(0, self._obs_noise_std, obs.shape).astype(np.float32)
        return obs

    def step(self, action):
        return super().step(action)


# Gymnasium registration (use envs.bark_ant_3leg when run with PYTHONPATH=bark repo root)
def register_bark_envs():
    from gymnasium.envs.registration import register
    register(
        id="BarkAnt3Leg-v0",
        entry_point="envs.bark_ant_3leg:BarkAnt3LegEnv",
        max_episode_steps=1000,
        kwargs={},
    )
