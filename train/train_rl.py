"""
Train an RL policy (PPO/SAC) on BarkAnt3Leg with optional W&B logging.
Usage:
  PYTHONPATH=. python -m train.train_rl --config configs/ppo_ant_3leg.yaml
  PYTHONPATH=. python -m train.train_rl --config configs/ppo_ant_3leg.yaml --wandb
"""
import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Register BARK envs before gymnasium.make
from envs import register_bark_envs

register_bark_envs()

import gymnasium as gym


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(
    env_id: str,
    env_kwargs: dict,
    seed: int = 0,
    amp_disc=None,
    amp_style_weight: float = 1.0,
):
    def _init():
        e = gym.make(env_id, **env_kwargs)
        e = Monitor(e)
        if amp_disc is not None:
            from envs.amp_wrapper import AMPRewardWrapper
            e = AMPRewardWrapper(e, amp_disc, style_weight=amp_style_weight)
        e.reset(seed=seed)
        return e
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_ant_3leg.yaml", help="Path to YAML config")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--comet", action="store_true", help="Enable Comet ML logging (set COMET_API_KEY)")
    parser.add_argument("--save_path", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--tb_dir", type=str, default="logs/tensorboard", help="TensorBoard log directory (set to empty to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps from config")
    args = parser.parse_args()
    _repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _repo_root / config_path
    config = load_config(str(config_path))
    env_id = config.get("env_id", "BarkAnt3Leg-v0")
    env_kwargs = config.get("env_kwargs", {})

    amp_config = config.get("amp") or {}
    amp_enabled = amp_config.get("enabled", False)
    amp_disc = None
    amp_trainer = None
    amp_style_weight = float(amp_config.get("style_weight", 1.0))

    if amp_enabled:
        expert_path = amp_config.get("expert_path")
        if not expert_path:
            raise ValueError("amp.enabled is true but amp.expert_path is not set")
        expert_path = Path(expert_path)
        if not expert_path.is_absolute():
            expert_path = _repo_root / expert_path
        if not expert_path.exists():
            raise FileNotFoundError(f"AMP expert path not found: {expert_path}")
        # Get obs_dim from env
        _tmp = gym.make(env_id, **env_kwargs)
        obs_dim = int(_tmp.observation_space.shape[0])
        _tmp.close()
        from train.amp import (
            AMPDiscriminator,
            AMPTrainer,
            load_reference_transitions,
        )
        ref_trans = load_reference_transitions(
            expert_path,
            obs_dim=obs_dim,
            max_transitions=amp_config.get("max_transitions"),
        )
        amp_disc = AMPDiscriminator(
            obs_dim,
            hidden_dims=tuple(amp_config.get("disc_hidden", [256, 256])),
            dropout=float(amp_config.get("disc_dropout", 0.1)),
        )
        amp_trainer = AMPTrainer(
            amp_disc,
            ref_trans,
            lr=float(amp_config.get("disc_lr", 1e-4)),
            batch_size=int(amp_config.get("disc_batch_size", 256)),
        )

    n_envs = 1
    vec_env = DummyVecEnv([
        make_env(
            env_id,
            env_kwargs,
            seed=args.seed + i,
            amp_disc=amp_disc,
            amp_style_weight=amp_style_weight,
        )
        for i in range(n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env(
            env_id,
            env_kwargs,
            seed=args.seed + 100,
            amp_disc=amp_disc,
            amp_style_weight=amp_style_weight,
        )
    ])
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    tb_log = None
    if args.tb_dir:
        tb_log = str(_repo_root / args.tb_dir)
        Path(tb_log).mkdir(parents=True, exist_ok=True)
    eval_freq = config.get("eval_freq", 5000)
    n_eval_episodes = config.get("n_eval_episodes", 5)
    save_freq = config.get("save_freq", 25000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    callbacks = [eval_callback]
    if amp_trainer is not None:
        from train.callbacks import AMPCallback
        callbacks.append(AMPCallback(amp_trainer, verbose=1))
    from train.callbacks import LegMetricsCallback
    callbacks.append(LegMetricsCallback(verbose=0))

    alg_name = config.get("algorithm", "PPO")
    run_name = config.get("run_name") or f"{config.get('algorithm', 'PPO')}_{env_id}_seed{args.seed}"
    algo_kwargs = {
        "env": vec_env,
        "seed": args.seed,
        "verbose": 1,
        "tensorboard_log": tb_log,
    }
    if alg_name == "PPO":
        algo_kwargs.update(
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
        )
        model = PPO("MlpPolicy", **algo_kwargs)
    elif alg_name == "SAC":
        algo_kwargs.update(
            learning_rate=config.get("learning_rate", 3e-4),
            buffer_size=config.get("buffer_size", 100_000),
            batch_size=config.get("batch_size", 256),
            gamma=config.get("gamma", 0.99),
        )
        model = SAC("MlpPolicy", **algo_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback as WandbSB3Callback

            run_name = config.get("run_name") or f"{alg_name}_{env_id}_seed{args.seed}"
            wandb.init(project=config.get("project_name", "bark-rl"), name=run_name, config=config)
            callbacks.append(
                WandbSB3Callback(
                    model_save_path=str(save_path / "wandb"),
                    model_save_freq=save_freq,
                    verbose=1,
                )
            )
        except ImportError:
            import wandb
            from stable_baselines3.common.callbacks import BaseCallback

            run_name = config.get("run_name") or f"{alg_name}_{env_id}_seed{args.seed}"
            wandb.init(project=config.get("project_name", "bark-rl"), name=run_name, config=config)

            class WandbLogCallback(BaseCallback):
                def _on_rollout_end(self) -> None:
                    if self.logger is None:
                        return
                    logs = {k: v for k, v in self.logger.name_to_value.items() if v is not None}
                    if logs:
                        wandb.log(logs, step=self.num_timesteps)

            callbacks.append(WandbLogCallback())

    if args.comet:
        try:
            from train.callbacks import CometLoggerCallback  # noqa: PLC0415
            callbacks.append(CometLoggerCallback(verbose=1))
        except Exception as e:
            print("Comet not available:", e)

    total_timesteps = args.timesteps if args.timesteps is not None else config.get("total_timesteps", 500_000)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(save_path / "final_model")
    vec_env.close()
    eval_env.close()
    if args.wandb:
        try:
            wandb.finish()
        except Exception:
            pass
    print("Training done. Model saved to", save_path)


if __name__ == "__main__":
    main()
