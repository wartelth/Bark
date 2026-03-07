"""
Spawn the Bark Ant quadruped in MuJoCo and run it with 3D visualization.
Use a trained policy (PPO/SAC) or random actions. Optional video recording.

Usage:
  PYTHONPATH=. python scripts/run_dog_viz.py
  PYTHONPATH=. python scripts/run_dog_viz.py --model models/best.zip --episodes 10
  PYTHONPATH=. python scripts/run_dog_viz.py --config configs/env_ant_3leg.yaml --record --video-folder logs/videos
"""
import argparse
from pathlib import Path

import yaml
import gymnasium as gym

from envs import register_bark_envs

register_bark_envs()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env_kwargs_from_env_config(config: dict) -> dict:
    """Build env_kwargs from env_ant_3leg-style config (obs_noise_std, prosthetic_leg_index, reward_*)."""
    kwargs = {}
    if "obs_noise_std" in config:
        kwargs["obs_noise_std"] = config["obs_noise_std"]
    if "prosthetic_leg_index" in config:
        kwargs["prosthetic_leg_index"] = config["prosthetic_leg_index"]
    # Optional AntEnv reward weights (Gymnasium Ant v4 names)
    if "reward_healthy" in config:
        kwargs["healthy_reward"] = config["reward_healthy"]
    if "reward_ctrl" in config:
        kwargs["ctrl_cost_weight"] = abs(config["reward_ctrl"])
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Run Bark Ant (dog) in MuJoCo with 3D visualization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_ant_3leg.yaml",
        help="Path to YAML config (ppo or env); used for env_id and env_kwargs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to SB3 model .zip (PPO or SAC). If not set, use random actions.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for env reset.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless (no 3D window); useful for policy evaluation only.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video to --video-folder (uses rgb_array render mode).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="logs/videos",
        help="Folder for recorded videos when --record is set.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = load_config(str(config_path))

    env_id = config.get("env_id", "BarkAnt3Leg-v0")
    env_kwargs = config.get("env_kwargs") or {}

    # If env_kwargs is empty, try loading env-specific config for obs_noise_std, etc.
    if not env_kwargs and "obs_noise_std" not in config:
        env_config_path = repo_root / "configs" / "env_ant_3leg.yaml"
        if env_config_path.exists():
            env_config = load_config(str(env_config_path))
            env_kwargs = build_env_kwargs_from_env_config(env_config)
    elif not env_kwargs:
        env_kwargs = build_env_kwargs_from_env_config(config)

    # Render mode
    if args.record:
        render_mode = "rgb_array"
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "human"
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)

    if args.record:
        from gymnasium.wrappers import RecordVideo
        video_path = Path(args.video_folder)
        video_path.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )

    # Load policy if requested
    model = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(model_path))
        except Exception:
            try:
                from stable_baselines3 import SAC
                model = SAC.load(str(model_path))
            except Exception as e:
                raise RuntimeError(
                    f"Could not load model as PPO or SAC: {e}"
                ) from e
        print(f"Loaded policy from {model_path}", flush=True)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            steps = 0
            while not done:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            print(f"Episode {ep + 1}/{args.episodes} finished in {steps} steps.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        env.close()
        if args.record:
            print(f"Videos saved to {args.video_folder}", flush=True)


if __name__ == "__main__":
    main()
