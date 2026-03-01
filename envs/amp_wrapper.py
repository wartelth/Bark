"""
Gymnasium wrapper that adds AMP style reward: r_style = w * max(0, 1 - 0.25*(D(s,s')-1)^2).
Requires a discriminator that implements predict_reward(s, s_next).
"""
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class AMPRewardWrapper(gym.Wrapper):
    """
    Adds adversarial motion prior reward to the environment reward.
    Stores previous observation to form (s, s') with current obs after step.
    """

    def __init__(
        self,
        env: gym.Env,
        discriminator: Any,
        style_weight: float = 1.0,
    ):
        super().__init__(env)
        self.discriminator = discriminator
        self.style_weight = style_weight
        self._prev_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._prev_obs is not None and self.discriminator is not None:
            r_style = self.discriminator.predict_reward(self._prev_obs, obs)
            if np.isscalar(r_style):
                reward = float(reward) + self.style_weight * float(r_style)
            else:
                reward = float(reward) + self.style_weight * float(r_style.squeeze())
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info
