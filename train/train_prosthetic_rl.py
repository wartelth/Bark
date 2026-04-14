"""
Train an RL student that controls only leg 3 in the ProstheticGo1 env.
Teacher handles legs 0-2. Student's 3D action = (hip, thigh, calf) of leg 3.

Usage:
    PYTHONPATH=. python train/train_prosthetic_rl.py
    PYTHONPATH=. python train/train_prosthetic_rl.py --config configs/prosthetic_rl_go1.yaml
"""
import argparse
from pathlib import Path

import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envs.prosthetic_env  # noqa: F401 (registers ProstheticGo1-v0)

REPO = Path(__file__).resolve().parent.parent
SAVE_DIR = REPO / "models" / "prosthetic_rl"


def make_env(
    teacher_path: str | None = None,
    obs_noise: float = 0.0,
    reward_tracking_weight: float = 1.0,
    reward_forward_weight: float = 1.0,
    reward_alive_weight: float = 0.0,
    scenario_pool: str = "all_train",
    fixed_scenario: str | None = None,
    mass_rand_pct: float = 0.0,
    friction_rand_pct: float = 0.0,
    seed: int = 0,
):
    def _init():
        kwargs = {
            "reward_tracking_weight": reward_tracking_weight,
            "reward_forward_weight": reward_forward_weight,
            "reward_alive_weight": reward_alive_weight,
            "scenario_pool": scenario_pool,
            "fixed_scenario": fixed_scenario,
            "mass_rand_pct": mass_rand_pct,
            "friction_rand_pct": friction_rand_pct,
        }
        if teacher_path:
            kwargs["teacher_model_path"] = teacher_path
        if obs_noise > 0:
            kwargs["obs_noise_std"] = obs_noise
        e = gym.make("ProstheticGo1-v0", **kwargs)
        e = Monitor(e)
        e.reset(seed=seed)
        return e
    return _init


def build_vec_env(
    n_envs: int,
    teacher: str | None,
    obs_noise: float,
    reward_tracking_weight: float,
    reward_forward_weight: float,
    reward_alive_weight: float,
    scenario_pool: str,
    fixed_scenario: str | None,
    mass_rand_pct: float,
    friction_rand_pct: float,
):
    env_fns = [
        make_env(
            teacher,
            obs_noise,
            reward_tracking_weight,
            reward_forward_weight,
            reward_alive_weight,
            scenario_pool=scenario_pool,
            fixed_scenario=fixed_scenario,
            mass_rand_pct=mass_rand_pct,
            friction_rand_pct=friction_rand_pct,
            seed=i,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        timesteps = args.timesteps if args.timesteps != parser.get_default("timesteps") else cfg.get("total_timesteps", args.timesteps)
        obs_noise = args.obs_noise if args.obs_noise != parser.get_default("obs_noise") else cfg.get("obs_noise_std", args.obs_noise)
        teacher = args.teacher if args.teacher is not None else cfg.get("teacher_model_path", args.teacher)
        reward_tracking_weight = cfg.get("reward_tracking_weight", 1.0)
        reward_forward_weight = cfg.get("reward_forward_weight", 1.0)
        reward_alive_weight = cfg.get("reward_alive_weight", 0.0)
        learning_rate = cfg.get("learning_rate", 3e-4)
        n_steps = cfg.get("n_steps", 2048)
        batch_size = cfg.get("batch_size", 64)
        n_epochs = cfg.get("n_epochs", 10)
        gamma = cfg.get("gamma", 0.99)
        gae_lambda = cfg.get("gae_lambda", 0.95)
        clip_range = cfg.get("clip_range", 0.2)
        ent_coef = cfg.get("ent_coef", 0.01)
        vf_coef = cfg.get("vf_coef", 0.5)
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        eval_freq = cfg.get("eval_freq", 25_000)
        n_eval_episodes = cfg.get("n_eval_episodes", 10)
        n_envs = args.n_envs if args.n_envs != parser.get_default("n_envs") else cfg.get("n_envs", args.n_envs)
        scenario_pool = cfg.get("scenario_pool", "all_train")
        eval_scenario_pool = cfg.get("eval_scenario_pool", scenario_pool)
        fixed_eval_scenario = cfg.get("fixed_eval_scenario")
        mass_rand_pct = cfg.get("mass_rand_pct", 0.0)
        friction_rand_pct = cfg.get("friction_rand_pct", 0.0)
    else:
        timesteps = args.timesteps
        obs_noise = args.obs_noise
        teacher = args.teacher
        reward_tracking_weight = 1.0
        reward_forward_weight = 1.0
        reward_alive_weight = 0.0
        learning_rate = 3e-4
        n_steps = 2048
        batch_size = 64
        n_epochs = 10
        gamma = 0.99
        gae_lambda = 0.95
        clip_range = 0.2
        ent_coef = 0.01
        vf_coef = 0.5
        max_grad_norm = 0.5
        eval_freq = 25_000
        n_eval_episodes = 10
        n_envs = args.n_envs
        scenario_pool = "all_train"
        eval_scenario_pool = scenario_pool
        fixed_eval_scenario = None
        mass_rand_pct = 0.0
        friction_rand_pct = 0.0

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    env = build_vec_env(
        n_envs=n_envs,
        teacher=teacher,
        obs_noise=obs_noise,
        reward_tracking_weight=reward_tracking_weight,
        reward_forward_weight=reward_forward_weight,
        reward_alive_weight=reward_alive_weight,
        scenario_pool=scenario_pool,
        fixed_scenario=None,
        mass_rand_pct=mass_rand_pct,
        friction_rand_pct=friction_rand_pct,
    )
    eval_env = DummyVecEnv([
        make_env(
            teacher,
            obs_noise,
            reward_tracking_weight,
            reward_forward_weight,
            reward_alive_weight,
            scenario_pool=eval_scenario_pool,
            fixed_scenario=fixed_eval_scenario,
            mass_rand_pct=0.0,
            friction_rand_pct=0.0,
            seed=99,
        )
    ])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        device=args.device,
        tensorboard_log=str(SAVE_DIR / "tb_logs"),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR),
        log_path=str(SAVE_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    print(f"Training prosthetic RL student for {timesteps} steps with n_envs={n_envs} on device={args.device}...")
    model.learn(total_timesteps=timesteps, callback=[eval_cb])
    model.save(str(SAVE_DIR / "prosthetic_rl_final"))
    env.close()
    eval_env.close()
    print(f"Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
