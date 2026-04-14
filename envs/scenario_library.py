from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    desired_velocity: tuple[float, float, float]
    slope_pitch_deg: float = 0.0
    weight: float = 1.0


SCENARIO_LIBRARY: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("chill_walk", (0.25, 0.0, 0.0), weight=1.0),
    ScenarioSpec("slow_walk", (0.35, 0.0, 0.0), weight=1.4),
    ScenarioSpec("medium_walk", (0.55, 0.0, 0.0), weight=1.6),
    ScenarioSpec("fast_walk", (0.68, 0.0, 0.0), weight=1.2),
    ScenarioSpec("run", (0.80, 0.0, 0.0), weight=1.0),
    ScenarioSpec("turn_left", (0.45, 0.0, 0.30), weight=0.8),
    ScenarioSpec("turn_right", (0.45, 0.0, -0.30), weight=0.8),
    ScenarioSpec("strafe_left_cmd", (0.40, 0.18, 0.0), weight=0.5),
    ScenarioSpec("strafe_right_cmd", (0.40, -0.18, 0.0), weight=0.5),
    ScenarioSpec("uphill_gentle", (0.45, 0.0, 0.0), slope_pitch_deg=4.0, weight=0.9),
    ScenarioSpec("uphill_medium", (0.40, 0.0, 0.0), slope_pitch_deg=8.0, weight=0.7),
    ScenarioSpec("downhill_gentle", (0.45, 0.0, 0.0), slope_pitch_deg=-4.0, weight=0.9),
    ScenarioSpec("downhill_medium", (0.40, 0.0, 0.0), slope_pitch_deg=-8.0, weight=0.7),
)


SCENARIO_POOLS: dict[str, tuple[str, ...]] = {
    "demo_supported": ("slow_walk", "medium_walk", "fast_walk", "run"),
    "flat_supported": ("chill_walk", "slow_walk", "medium_walk", "fast_walk", "run"),
    "flat_plus_turns": (
        "chill_walk",
        "slow_walk",
        "medium_walk",
        "fast_walk",
        "run",
        "turn_left",
        "turn_right",
    ),
    "all_train": tuple(spec.name for spec in SCENARIO_LIBRARY),
    "slopes_only": ("uphill_gentle", "uphill_medium", "downhill_gentle", "downhill_medium"),
}


def scenario_by_name(name: str) -> ScenarioSpec:
    for spec in SCENARIO_LIBRARY:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown scenario: {name}")


def scenario_pool(pool_name: str) -> list[ScenarioSpec]:
    if pool_name in SCENARIO_POOLS:
        return [scenario_by_name(name) for name in SCENARIO_POOLS[pool_name]]
    return [scenario_by_name(name.strip()) for name in pool_name.split(",") if name.strip()]


def _rng_choice(rng, n: int, probs: np.ndarray) -> int:
    if hasattr(rng, "choice"):
        return int(rng.choice(n, p=probs))
    return int(np.random.choice(n, p=probs))


def _rng_uniform(rng, low: float, high: float, shape) -> np.ndarray:
    if hasattr(rng, "uniform"):
        return rng.uniform(low, high, shape)
    return np.random.uniform(low, high, shape)


def sample_scenario(rng, pool_name: str = "all_train") -> ScenarioSpec:
    pool = scenario_pool(pool_name)
    weights = np.array([spec.weight for spec in pool], dtype=np.float64)
    probs = weights / weights.sum()
    idx = _rng_choice(rng, len(pool), probs)
    return pool[idx]


def desired_velocity_array(spec: ScenarioSpec) -> np.ndarray:
    return np.array(spec.desired_velocity, dtype=np.float32)


def _pitch_quaternion(pitch_deg: float) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    return np.array([math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0], dtype=np.float64)


def apply_scenario(
    env,
    spec: ScenarioSpec,
    rng: np.random.RandomState | None = None,
    mass_rand_pct: float = 0.0,
    friction_rand_pct: float = 0.0,
):
    base = env.unwrapped
    desired_velocity = desired_velocity_array(spec)
    base._desired_velocity_min = desired_velocity.copy()
    base._desired_velocity_max = desired_velocity.copy()
    base._desired_velocity = desired_velocity.copy()

    if not hasattr(base, "_bark_floor_geom_id"):
        base._bark_floor_geom_id = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM.value, "floor")
    if not hasattr(base, "_bark_orig_floor_quat"):
        base._bark_orig_floor_quat = base.model.geom_quat[base._bark_floor_geom_id].copy()
    if not hasattr(base, "_bark_orig_body_mass"):
        base._bark_orig_body_mass = base.model.body_mass.copy()
    if not hasattr(base, "_bark_orig_geom_friction"):
        base._bark_orig_geom_friction = base.model.geom_friction.copy()

    floor_id = base._bark_floor_geom_id
    base.model.geom_quat[floor_id] = base._bark_orig_floor_quat
    if abs(spec.slope_pitch_deg) > 1e-6:
        base.model.geom_quat[floor_id] = _pitch_quaternion(spec.slope_pitch_deg)

    base.model.body_mass[:] = base._bark_orig_body_mass
    base.model.geom_friction[:] = base._bark_orig_geom_friction

    if rng is not None and mass_rand_pct > 0:
        mass_noise = _rng_uniform(rng, 1 - mass_rand_pct, 1 + mass_rand_pct, base.model.body_mass.shape)
        base.model.body_mass[:] = base.model.body_mass * mass_noise

    if rng is not None and friction_rand_pct > 0:
        friction_noise = _rng_uniform(rng, 1 - friction_rand_pct, 1 + friction_rand_pct, base.model.geom_friction.shape)
        base.model.geom_friction[:] = base.model.geom_friction * friction_noise


def scenario_table(pool_name: str) -> list[dict]:
    return [
        {
            "name": spec.name,
            "desired_velocity": list(spec.desired_velocity),
            "slope_pitch_deg": spec.slope_pitch_deg,
            "weight": spec.weight,
        }
        for spec in scenario_pool(pool_name)
    ]


def scenario_names(specs: Iterable[ScenarioSpec]) -> list[str]:
    return [spec.name for spec in specs]
