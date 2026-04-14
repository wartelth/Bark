"""
ProstheticGo1Env built around the working external Go1 teacher/environment.

Teacher controls legs 0-2, student controls leg 3 only. The only purpose of this
env is to compare supervised imitation vs RL on following a teacher's leg-3 behavior.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pretrained.load_teacher import GO1_LEG3_JOINT_IX, LEG3_OBS_REMOVE_IX, load_teacher, make_go1_env
from envs.scenario_library import apply_scenario, sample_scenario, scenario_by_name


class ProstheticGo1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        teacher_model_path: Optional[str] = None,
        obs_noise_std: float = 0.0,
        render_mode: Optional[str] = None,
        reward_tracking_weight: float = 1.0,
        reward_forward_weight: float = 1.0,
        reward_alive_weight: float = 0.0,
        scenario_pool: str = "all_train",
        fixed_scenario: Optional[str] = None,
        mass_rand_pct: float = 0.0,
        friction_rand_pct: float = 0.0,
    ):
        super().__init__()

        self._teacher = load_teacher(teacher_model_path)
        self._obs_noise_std = obs_noise_std
        self._reward_tracking_weight = reward_tracking_weight
        self._reward_forward_weight = reward_forward_weight
        self._reward_alive_weight = reward_alive_weight
        self._scenario_pool = scenario_pool
        self._fixed_scenario = fixed_scenario
        self._mass_rand_pct = mass_rand_pct
        self._friction_rand_pct = friction_rand_pct
        self._current_scenario_name = fixed_scenario or "unknown"
        self._current_slope_pitch_deg = 0.0

        self._inner = make_go1_env(render_mode=render_mode)
        self._remove_ix = LEG3_OBS_REMOVE_IX

        full_obs_dim = int(self._inner.observation_space.shape[0])
        self._obs_3leg_dim = full_obs_dim - len(self._remove_ix)
        action_low = self._inner.action_space.low[GO1_LEG3_JOINT_IX].astype(np.float32)
        action_high = self._inner.action_space.high[GO1_LEG3_JOINT_IX].astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_3leg_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self._full_obs = None

    def _get_3leg_obs(self, full_obs: np.ndarray) -> np.ndarray:
        obs_3leg = np.delete(full_obs, self._remove_ix).astype(np.float32)
        if self._obs_noise_std > 0:
            obs_3leg += self.np_random.normal(0, self._obs_noise_std, obs_3leg.shape).astype(np.float32)
        return obs_3leg

    def reset(self, seed=None, options=None):
        if self._fixed_scenario:
            spec = scenario_by_name(self._fixed_scenario)
        else:
            spec = sample_scenario(self.np_random, self._scenario_pool)
        apply_scenario(
            self._inner,
            spec,
            self.np_random,
            mass_rand_pct=self._mass_rand_pct,
            friction_rand_pct=self._friction_rand_pct,
        )
        self._full_obs, info = self._inner.reset(seed=seed, options=options)
        self._current_scenario_name = spec.name
        self._current_slope_pitch_deg = float(spec.slope_pitch_deg)
        return self._get_3leg_obs(self._full_obs), info

    def step(self, student_action: np.ndarray):
        teacher_action_full, _ = self._teacher.predict(self._full_obs, deterministic=True)
        teacher_leg3 = teacher_action_full[GO1_LEG3_JOINT_IX].copy()

        combined_action = np.array(teacher_action_full, copy=True)
        combined_action[GO1_LEG3_JOINT_IX] = student_action

        self._full_obs, base_reward, terminated, truncated, info = self._inner.step(combined_action)

        tracking_error = float(np.mean((student_action - teacher_leg3) ** 2))
        reward = (
            self._reward_forward_weight * float(base_reward)
            - self._reward_tracking_weight * tracking_error
            + self._reward_alive_weight * (1.0 if not terminated else 0.0)
        )

        info["tracking_mse"] = tracking_error
        info["teacher_leg3"] = teacher_leg3
        info["student_leg3"] = np.asarray(student_action, dtype=np.float32).copy()
        info["scenario_name"] = self._current_scenario_name
        info["slope_pitch_deg"] = self._current_slope_pitch_deg

        return self._get_3leg_obs(self._full_obs), reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()


def register_prosthetic_env():
    from gymnasium.envs.registration import register
    env_id = "ProstheticGo1-v0"
    if env_id not in gym.envs.registry:
        register(
            id=env_id,
            entry_point="envs.prosthetic_env:ProstheticGo1Env",
            max_episode_steps=1000,
        )


register_prosthetic_env()
