"""
Record the Go1 PPO/SAC teacher walking (all 12 joints) to an MP4, same pipeline as render_students.

Produces:
  reports/walk_teacher.mp4

Usage:
    PYTHONPATH=. python -m postpro.render_teacher
    PYTHONPATH=. python -m postpro.render_teacher --search-seeds 40
    PYTHONPATH=. python -m postpro.render_teacher --steps 600 --fps 30 --out reports
    PYTHONPATH=. python -m postpro.render_teacher --teacher path/to/best_model.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pretrained.load_teacher import load_teacher, make_go1_env


def _make_env_offscreen():
    return make_go1_env(render_mode="rgb_array")


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30):
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


def _burn_label(frame: np.ndarray, text: str, x: int, y: int):
    try:
        import cv2
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
    except ImportError:
        pass


def find_longest_first_episode_seed(teacher, n_seeds: int, max_steps: int = 1000) -> tuple[int, int]:
    """Pick reset seed that yields the longest first episode (more likely to show locomotion)."""
    best_seed, best_len = 0, -1
    for s in range(n_seeds):
        env = make_go1_env(render_mode=None)
        obs, _ = env.reset(seed=s)
        steps = 0
        while steps < max_steps:
            a, _ = teacher.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            if term or trunc:
                break
        env.close()
        if steps > best_len:
            best_len = steps
            best_seed = s
    return best_seed, best_len


def rollout_teacher(teacher, env, n_steps: int, reset_seed: int) -> tuple[list[np.ndarray], float]:
    obs, _ = env.reset(seed=reset_seed)
    frames = []
    total_r = 0.0
    for step in range(n_steps):
        action, _ = teacher.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total_r += float(r)
        frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
        _burn_label(frame, "Teacher (4 legs, Go1)", 10, 28)
        _burn_label(frame, f"step {step}  seed={reset_seed}", 10, frame.shape[0] - 12)
        frames.append(frame)
        if term or trunc:
            obs, _ = env.reset(seed=step + 10_000)
    return frames, total_r


def main():
    parser = argparse.ArgumentParser(description="Render Go1 teacher walking to MP4")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="reports")
    parser.add_argument("--teacher", type=str, default=None, help="Path to best_model.zip (default: pretrained/go1_teacher)")
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=0,
        metavar="N",
        help="Try seeds 0..N-1 and record using the seed with the longest first episode (better demos).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    print("Loading teacher...")
    teacher = load_teacher(args.teacher)

    reset_seed = 0
    if args.search_seeds > 0:
        print(f"Searching seeds 0..{args.search_seeds - 1} for longest first episode...")
        reset_seed, ep_len = find_longest_first_episode_seed(teacher, args.search_seeds)
        print(f"  Using seed={reset_seed} (first episode length={ep_len} steps)")

    env = _make_env_offscreen()
    print(f"Recording {args.steps} steps @ {args.fps} fps...")
    frames, total_r = rollout_teacher(teacher, env, args.steps, reset_seed=reset_seed)
    env.close()

    out_path = out_dir / "walk_teacher.mp4"
    print("Encoding video...")
    write_mp4(frames, out_path, fps=args.fps)
    print(f"Episode return (sum over steps, may span resets): {total_r:.1f}")
    print("Teacher video written successfully.")


if __name__ == "__main__":
    main()
