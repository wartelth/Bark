"""
Render teacher vs supervised vs IL vs RL under fixed command scenarios.

Produces truthful scenario videos for the commands this teacher actually follows well:

  reports/
    walk_teacher.mp4
    walk_supervised.mp4
    walk_il.mp4
    walk_rl.mp4
    walk_sidebyside.mp4
    walk_slow_sidebyside.mp4
    walk_fast_sidebyside.mp4
    walk_run_sidebyside.mp4
    leg3_action_traces.png

Usage:
    PYTHONPATH=. python -m postpro.render_students
    PYTHONPATH=. python -m postpro.render_students --scenario slow
    PYTHONPATH=. python -m postpro.render_students --all-scenarios
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrained.load_teacher import load_teacher, make_go1_env, split_obs_and_action, GO1_LEG3_JOINT_IX
from train.train_supervised import ProstheticMLP

REPO = Path(__file__).resolve().parent.parent
JOINT_NAMES = ["hip", "thigh", "calf"]
SCENARIOS = {
    "slow": {
        "title": "Slow walk",
        "velocity": np.array([0.35, 0.00, 0.00], dtype=np.float32),
        "prefix": "walk_slow",
    },
    "forward": {
        "title": "Medium walk",
        "velocity": np.array([0.55, 0.00, 0.00], dtype=np.float32),
        "prefix": "walk",
    },
    "fast": {
        "title": "Fast walk",
        "velocity": np.array([0.68, 0.00, 0.00], dtype=np.float32),
        "prefix": "walk_fast",
    },
    "run": {
        "title": "Run",
        "velocity": np.array([0.80, 0.00, 0.00], dtype=np.float32),
        "prefix": "walk_run",
    },
}


def _make_env_offscreen():
    return make_go1_env(render_mode="rgb_array")


def _set_command(env, desired_velocity: np.ndarray):
    base = env.unwrapped
    base._desired_velocity_min = desired_velocity.copy()
    base._desired_velocity_max = desired_velocity.copy()
    base._desired_velocity = desired_velocity.copy()


def _reset_with_command(env, desired_velocity: np.ndarray, seed: int):
    _set_command(env, desired_velocity)
    obs, info = env.reset(seed=seed)
    _set_command(env, desired_velocity)
    return obs, info


def _load_checkpoint_model(model_dir: Path):
    env = _make_env_offscreen()
    obs, _ = env.reset(seed=0)
    teacher = load_teacher()
    action, _ = teacher.predict(obs, deterministic=True)
    obs_3leg, _ = split_obs_and_action(obs, action)
    obs_dim = obs_3leg.shape[0]
    env.close()

    model = ProstheticMLP(obs_dim, action_dim=3)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _load_supervised():
    return _load_checkpoint_model(REPO / "models" / "supervised_prosthetic")


def _load_il():
    return _load_checkpoint_model(REPO / "models" / "imitation_prosthetic")


def _load_rl():
    from stable_baselines3 import PPO
    rl_path = REPO / "models" / "prosthetic_rl" / "prosthetic_rl_final.zip"
    return PPO.load(str(rl_path), device="cpu")


def rollout_teacher(teacher, env, n_steps: int, label: str, desired_velocity: np.ndarray, reset_seed: int = 0):
    obs, _ = _reset_with_command(env, desired_velocity, seed=reset_seed)
    frames = []
    rewards = []

    for step in range(n_steps):
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(teacher_action)
        frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
        _burn_label(frame, label, x=10, y=25)
        _burn_label(frame, f"cmd={desired_velocity.tolist()} step={step}", x=10, y=frame.shape[0] - 12)
        frames.append(frame)
        rewards.append(r)
        if term or trunc:
            obs, _ = _reset_with_command(env, desired_velocity, seed=reset_seed + step + 1)

    return {
        "frames": frames,
        "rewards": np.array(rewards),
        "label": label.lower(),
    }


def rollout_hybrid(teacher, student_fn, env, n_steps: int, label: str, desired_velocity: np.ndarray, reset_seed: int = 0):
    """
    Run hybrid rollout: teacher legs 0-2, student leg 3.
    Returns frames, per-step teacher leg3 actions, student leg3 actions, and rewards.
    """
    obs, _ = _reset_with_command(env, desired_velocity, seed=reset_seed)
    frames = []
    teacher_leg3_trace = []
    student_leg3_trace = []
    rewards = []

    for step in range(n_steps):
        teacher_action, _ = teacher.predict(obs, deterministic=True)
        obs_3leg, teacher_leg3 = split_obs_and_action(obs, teacher_action)

        student_leg3 = student_fn(obs_3leg)

        combined = teacher_action.copy()
        combined[GO1_LEG3_JOINT_IX] = student_leg3

        obs, r, term, trunc, _ = env.step(combined)
        frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
        _burn_label(frame, label, x=10, y=25)
        _burn_label(frame, f"cmd={desired_velocity.tolist()} step={step}", x=10, y=frame.shape[0] - 12)
        frames.append(frame)
        teacher_leg3_trace.append(teacher_leg3.copy())
        student_leg3_trace.append(student_leg3.copy())
        rewards.append(r)

        if term or trunc:
            obs, _ = _reset_with_command(env, desired_velocity, seed=reset_seed + step + 1)

    return {
        "frames": frames,
        "teacher_leg3": np.array(teacher_leg3_trace),
        "student_leg3": np.array(student_leg3_trace),
        "rewards": np.array(rewards),
        "label": label,
    }


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30):
    """Write frames to MP4 using pyav directly."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    for frame_arr in frames:
        av_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved {path} ({len(frames)} frames, {len(frames)/fps:.1f}s, {size_mb:.1f}MB)")


def stitch_panels(panels: list[tuple[list[np.ndarray], str]]) -> list[np.ndarray]:
    """Stitch multiple frame sequences horizontally with labels."""
    n = min(len(frames) for frames, _ in panels)
    stitched = []

    for i in range(n):
        frame_samples = [frames[i] for frames, _ in panels]
        labels = [label for _, label in panels]
        h = max(frame.shape[0] for frame in frame_samples)
        divider = 4
        w_total = sum(frame.shape[1] for frame in frame_samples) + divider * (len(frame_samples) - 1)
        canvas = np.zeros((h, w_total, 3), dtype=np.uint8)
        cursor = 0
        for idx, frame in enumerate(frame_samples):
            canvas[:frame.shape[0], cursor:cursor + frame.shape[1]] = frame
            _burn_label(canvas, labels[idx], x=cursor + 10, y=25)
            cursor += frame.shape[1]
            if idx < len(frame_samples) - 1:
                canvas[:, cursor:cursor + divider] = 60
                cursor += divider
        step_text = f"step {i}"
        _burn_label(canvas, step_text, x=w_total // 2 - 30, y=h - 10)
        stitched.append(canvas)

    return stitched


def _burn_label(frame: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255)):
    """Burn text onto frame using simple pixel font (no PIL dependency)."""
    try:
        import cv2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Action trace plot
# ---------------------------------------------------------------------------

def plot_leg3_traces(
    rollouts: list[dict],
    out_dir: Path,
    filename: str = "leg3_action_traces.png",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    style_map = {
        "supervised": {"color": "#2196F3", "ls": "-"},
        "il": {"color": "#8E24AA", "ls": "-"},
        "rl": {"color": "#F44336", "ls": "-"},
    }
    teacher_plotted = False

    for rollout in rollouts:
        label = rollout["label"].lower()
        style = style_map.get(label, {"color": "gray", "ls": "-"})

        teacher_trace = rollout["teacher_leg3"]
        student_trace = rollout["student_leg3"]
        steps = np.arange(len(student_trace))

        for j in range(3):
            ax = axes[j]

            if not teacher_plotted:
                ax.plot(steps, teacher_trace[:, j], color="black", linewidth=1.5,
                        alpha=0.5, linestyle="--", label="teacher (target)")

            ax.plot(steps, student_trace[:, j], color=style["color"],
                    linewidth=1.8, linestyle=style["ls"], alpha=0.85,
                    label=f"{label} student")

            error = np.abs(student_trace[:, j] - teacher_trace[:, j])
            ax.fill_between(steps, teacher_trace[:, j] - error, teacher_trace[:, j] + error,
                            color=style["color"], alpha=0.08)

        teacher_plotted = True

    for j, joint in enumerate(JOINT_NAMES):
        axes[j].set_ylabel(f"Leg 3 {joint}")
        axes[j].legend(fontsize=8, loc="upper right")
        axes[j].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation Step")
    axes[0].set_title("Leg 3 Joint Actions: Student Models vs Teacher Target")

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_tracking_error(
    rollouts: list[dict],
    out_dir: Path,
    filename: str = "leg3_tracking_error.png",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    colors = {"supervised": "#2196F3", "il": "#8E24AA", "rl": "#F44336"}

    for rollout in rollouts:
        label = rollout["label"].lower()
        color = colors.get(label, "gray")
        teacher = rollout["teacher_leg3"]
        student = rollout["student_leg3"]
        steps = np.arange(len(student))

        per_step_mse = np.mean((student - teacher) ** 2, axis=1)
        cumulative_mse = np.cumsum(per_step_mse) / (steps + 1)

        axes[0].plot(steps, per_step_mse, color=color, alpha=0.3, linewidth=0.8)
        window = min(20, len(per_step_mse))
        if len(per_step_mse) > window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(per_step_mse, kernel, mode="valid")
            axes[0].plot(steps[:len(smoothed)], smoothed, color=color,
                         linewidth=2, label=f"{label} (smoothed)")
        else:
            axes[0].plot(steps, per_step_mse, color=color, linewidth=2, label=label)

        axes[1].plot(steps, cumulative_mse, color=color, linewidth=2, label=label)

    axes[0].set_ylabel("Per-Step MSE")
    axes[0].set_title("Tracking Error: Student vs Teacher Leg 3 Actions")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Cumulative Mean MSE")
    axes[1].set_xlabel("Simulation Step")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


def plot_reward_comparison(
    rollouts: list[dict],
    out_dir: Path,
    filename: str = "leg3_reward_comparison.png",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"supervised": "#2196F3", "il": "#8E24AA", "rl": "#F44336"}

    for rollout in rollouts:
        label = rollout["label"].lower()
        color = colors.get(label, "gray")
        rewards = rollout["rewards"]
        cumulative = np.cumsum(rewards)
        steps = np.arange(len(rewards))
        ax.plot(steps, cumulative, color=color, linewidth=2, label=f"{label} (total: {cumulative[-1]:.1f})")

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Hybrid Walking: Cumulative Reward by Student")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _render_scenario(
    teacher,
    sup_fn,
    il_fn,
    rl_fn,
    out_dir: Path,
    scenario_key: str,
    n_steps: int,
    fps: int,
    write_videos: bool,
    write_individual_forward_assets: bool,
):
    scenario = SCENARIOS[scenario_key]
    desired_velocity = scenario["velocity"]
    prefix = scenario["prefix"]
    title = scenario["title"]

    print(f"\n=== Scenario: {title} ({desired_velocity.tolist()}) ===")

    env_teacher = _make_env_offscreen()
    roll_teacher = rollout_teacher(teacher, env_teacher, n_steps, "Teacher", desired_velocity)
    env_teacher.close()

    env_sup = _make_env_offscreen()
    roll_sup = rollout_hybrid(teacher, sup_fn, env_sup, n_steps, "Supervised", desired_velocity)
    env_sup.close()

    roll_il = None
    if il_fn is not None:
        env_il = _make_env_offscreen()
        roll_il = rollout_hybrid(teacher, il_fn, env_il, n_steps, "IL", desired_velocity)
        env_il.close()

    env_rl = _make_env_offscreen()
    roll_rl = rollout_hybrid(teacher, rl_fn, env_rl, n_steps, "RL", desired_velocity)
    env_rl.close()

    if write_videos:
        panels = [
            (roll_teacher["frames"], "Teacher"),
            (roll_sup["frames"], "Supervised"),
        ]
        if roll_il is not None:
            panels.append((roll_il["frames"], "IL"))
        panels.append((roll_rl["frames"], "RL"))
        sidebyside = stitch_panels(panels)
        write_mp4(sidebyside, out_dir / f"{prefix}_sidebyside.mp4", fps=fps)

        if write_individual_forward_assets:
            write_mp4(roll_teacher["frames"], out_dir / "walk_teacher.mp4", fps=fps)
            write_mp4(roll_sup["frames"], out_dir / "walk_supervised.mp4", fps=fps)
            if roll_il is not None:
                write_mp4(roll_il["frames"], out_dir / "walk_il.mp4", fps=fps)
            write_mp4(roll_rl["frames"], out_dir / "walk_rl.mp4", fps=fps)
            write_mp4(sidebyside, out_dir / "walk_sidebyside.mp4", fps=fps)

    rollouts = [roll_sup]
    if roll_il is not None:
        rollouts.append(roll_il)
    rollouts.append(roll_rl)
    return rollouts


def main():
    parser = argparse.ArgumentParser(description="Render teacher/supervised/RL scenarios + compare leg 3 actions")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="reports")
    parser.add_argument("--no-video", action="store_true", help="Skip video, only generate plots")
    parser.add_argument("--scenario", type=str, default="forward", choices=sorted(SCENARIOS))
    parser.add_argument("--all-scenarios", action="store_true", help="Render slow, medium, fast, and run triptychs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading teacher...")
    teacher = load_teacher()

    print("Loading supervised student...")
    sup_model = _load_supervised()
    def sup_fn(obs_3leg):
        with torch.no_grad():
            t = torch.from_numpy(obs_3leg).unsqueeze(0)
            return sup_model(t).squeeze(0).numpy()

    il_model = None
    il_path = REPO / "models" / "imitation_prosthetic" / "best_model.pt"
    if il_path.exists():
        print("Loading IL student...")
        il_model = _load_il()

    print("Loading RL student...")
    rl_model = _load_rl()
    def rl_fn(obs_3leg):
        action, _ = rl_model.predict(obs_3leg, deterministic=True)
        return action

    il_fn = None
    if il_model is not None:
        def il_fn(obs_3leg):
            with torch.no_grad():
                t = torch.from_numpy(obs_3leg).unsqueeze(0)
                return il_model(t).squeeze(0).numpy()

    scenario_keys = list(SCENARIOS) if args.all_scenarios else [args.scenario]
    main_rollouts = None
    for scenario_key in scenario_keys:
        rollouts = _render_scenario(
            teacher=teacher,
            sup_fn=sup_fn,
            il_fn=il_fn,
            rl_fn=rl_fn,
            out_dir=out_dir,
            scenario_key=scenario_key,
            n_steps=args.steps,
            fps=args.fps,
            write_videos=not args.no_video,
            write_individual_forward_assets=not args.no_video and scenario_key == "forward",
        )
        if scenario_key == "forward":
            main_rollouts = rollouts

    if main_rollouts is None:
        desired_velocity = SCENARIOS[args.scenario]["velocity"]
        env_sup = _make_env_offscreen()
        roll_sup = rollout_hybrid(teacher, sup_fn, env_sup, args.steps, "supervised", desired_velocity)
        env_sup.close()
        rollouts = [roll_sup]
        if il_fn is not None:
            env_il = _make_env_offscreen()
            roll_il = rollout_hybrid(teacher, il_fn, env_il, args.steps, "il", desired_velocity)
            env_il.close()
            rollouts.append(roll_il)
        env_rl = _make_env_offscreen()
        roll_rl = rollout_hybrid(teacher, rl_fn, env_rl, args.steps, "rl", desired_velocity)
        env_rl.close()
        rollouts.append(roll_rl)
        main_rollouts = rollouts

    # --- Plots ---
    print("\nGenerating action trace plots...")
    rollouts = main_rollouts
    plot_leg3_traces(rollouts, out_dir)
    plot_tracking_error(rollouts, out_dir)
    plot_reward_comparison(rollouts, out_dir)

    # --- Summary stats ---
    print("\n" + "=" * 60)
    print("  Rollout Summary")
    print("=" * 60)
    for roll in rollouts:
        mse = np.mean((roll["student_leg3"] - roll["teacher_leg3"]) ** 2)
        print(f"  {roll['label']:>12}: reward={roll['rewards'].sum():.1f}, "
              f"mean MSE={mse:.6f}, "
              f"steps={len(roll['rewards'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
