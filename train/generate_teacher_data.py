"""
Run the trained Go1 teacher policy and record (3-leg obs, leg-3 action) pairs.

This version intentionally generates *diverse* teacher data across speeds, turn
commands, and slope scenarios so supervised / IL / RL are not trained on a narrow
flat-ground gait only.

Output: data/teacher_rollouts.npz with keys:
  - obs_3leg
  - action_leg3
  - action_full
  - desired_velocity
  - scenario_id
  - scenario_name
  - slope_pitch_deg
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrained.load_teacher import load_teacher, make_go1_env, split_obs_and_action
from envs.scenario_library import apply_scenario, sample_scenario, scenario_pool

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data"


def generate(
    model_path: str | None = None,
    total_steps: int = 1_000_000,
    obs_noise_std: float = 0.0,
    mass_rand_pct: float = 0.0,
    friction_rand_pct: float = 0.0,
    seed: int = 42,
    scenario_pool_name: str = "all_train",
):
    teacher = load_teacher(model_path)
    env = make_go1_env(render=False)

    rng = np.random.RandomState(seed)
    all_obs_3leg = []
    all_action_leg3 = []
    all_action_full = []
    all_desired_velocity = []
    all_scenario_id = []
    all_scenario_name = []
    all_slope_pitch_deg = []
    pool = scenario_pool(scenario_pool_name)

    scenario = sample_scenario(rng, scenario_pool_name)
    scenario_idx = next(i for i, spec in enumerate(pool) if spec.name == scenario.name)
    desired_velocity = np.array(scenario.desired_velocity, dtype=np.float32)
    apply_scenario(env, scenario, rng, mass_rand_pct, friction_rand_pct)
    obs, _ = env.reset(seed=seed)
    episodes = 0
    ep_reward = 0.0

    for step in trange(total_steps, desc="Generating teacher data"):
        action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, action_leg3 = split_obs_and_action(obs, action)

        if obs_noise_std > 0:
            obs_3leg = obs_3leg + rng.normal(0, obs_noise_std, obs_3leg.shape).astype(np.float32)

        all_obs_3leg.append(obs_3leg)
        all_action_leg3.append(action_leg3)
        all_action_full.append(action.astype(np.float32))
        all_desired_velocity.append(desired_velocity.copy())
        all_scenario_id.append(scenario_idx)
        all_scenario_name.append(scenario.name)
        all_slope_pitch_deg.append(float(scenario.slope_pitch_deg))

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            episodes += 1
            if episodes % 100 == 0:
                print(f"  Episode {episodes}, last reward: {ep_reward:.1f}")
            ep_reward = 0.0
            scenario = sample_scenario(rng, scenario_pool_name)
            desired_velocity = np.array(scenario.desired_velocity, dtype=np.float32)
            scenario_idx = next(i for i, spec in enumerate(pool) if spec.name == scenario.name)
            apply_scenario(env, scenario, rng, mass_rand_pct, friction_rand_pct)
            obs, _ = env.reset(seed=seed + episodes)

    env.close()

    obs_3leg_arr = np.array(all_obs_3leg, dtype=np.float32)
    action_leg3_arr = np.array(all_action_leg3, dtype=np.float32)
    action_full_arr = np.array(all_action_full, dtype=np.float32)
    desired_velocity_arr = np.array(all_desired_velocity, dtype=np.float32)
    scenario_id_arr = np.array(all_scenario_id, dtype=np.int32)
    scenario_name_arr = np.array(all_scenario_name)
    slope_pitch_deg_arr = np.array(all_slope_pitch_deg, dtype=np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "teacher_rollouts.npz"
    np.savez_compressed(
        out_path,
        obs_3leg=obs_3leg_arr,
        action_leg3=action_leg3_arr,
        action_full=action_full_arr,
        desired_velocity=desired_velocity_arr,
        scenario_id=scenario_id_arr,
        scenario_name=scenario_name_arr,
        slope_pitch_deg=slope_pitch_deg_arr,
    )
    print(f"\nSaved {len(obs_3leg_arr)} transitions to {out_path}")
    print(f"  obs_3leg shape: {obs_3leg_arr.shape}")
    print(f"  action_leg3 shape: {action_leg3_arr.shape}")
    print(f"  action_full shape: {action_full_arr.shape}")
    uniq, counts = np.unique(scenario_name_arr, return_counts=True)
    print("  scenario counts:")
    for name, count in zip(uniq, counts):
        print(f"    {name}: {int(count)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--noise", type=float, default=0.0, help="Obs noise std for domain rand")
    parser.add_argument("--mass-rand", type=float, default=0.0, help="Mass randomization %%")
    parser.add_argument("--friction-rand", type=float, default=0.0, help="Friction randomization %%")
    parser.add_argument("--scenario-pool", type=str, default="all_train", help="Scenario pool name or comma-separated scenario names")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        model_path=args.model,
        total_steps=args.steps,
        obs_noise_std=args.noise,
        mass_rand_pct=args.mass_rand,
        friction_rand_pct=args.friction_rand,
        seed=args.seed,
        scenario_pool_name=args.scenario_pool,
    )


if __name__ == "__main__":
    main()
