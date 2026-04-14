from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


REPO = Path(__file__).resolve().parent.parent


def _box(ax, xy, w, h, text, facecolor, edgecolor="#1f1f1f", fontsize=11, weight="bold"):
    rect = Rectangle(xy, w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=2, joinstyle="round")
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return rect


def _arrow(ax, p1, p2, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=color,
        )
    )


def draw_supervised_mlp(out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    colors = {
        "input": "#DCEEFF",
        "hidden": "#DFF3E4",
        "output": "#FFE1D6",
        "meta": "#F5F5F5",
    }

    x_positions = [0.6, 3.0, 5.7, 8.4, 11.2]
    widths = [1.6, 1.8, 1.8, 1.8, 1.6]
    labels = [
        "Student Input\n39 dims",
        "Linear\n39 -> 256\nReLU",
        "Linear\n256 -> 256\nReLU",
        "Linear\n256 -> 128\nReLU",
        "Output\n3 dims\nleg-3 action",
    ]
    fills = [colors["input"], colors["hidden"], colors["hidden"], colors["hidden"], colors["output"]]

    for idx, x in enumerate(x_positions):
        _box(ax, (x, 1.45), widths[idx], 1.1, labels[idx], fills[idx])
        if idx < len(x_positions) - 1:
            _arrow(ax, (x + widths[idx], 2.0), (x_positions[idx + 1] - 0.1, 2.0))

    _box(
        ax,
        (2.6, 0.25),
        8.0,
        0.7,
        "Behavior cloning / supervised imitation: minimize MSE between predicted leg-3 action and teacher leg-3 action",
        colors["meta"],
        fontsize=10,
        weight="normal",
    )
    ax.set_title("Supervised Student MLP", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "supervised_mlp_diagram.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_student_io(out_dir: Path):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    colors = {
        "teacher": "#FDE2E4",
        "obs": "#DCEEFF",
        "mask": "#FFF2CC",
        "student": "#DFF3E4",
        "output": "#FFE1D6",
    }

    _box(ax, (0.7, 5.3), 2.6, 1.1, "Teacher policy", colors["teacher"])
    _box(ax, (0.7, 2.0), 2.6, 1.4, "Go1 env\n48-dim observation", colors["obs"])

    _box(ax, (4.2, 4.9), 3.4, 1.9, "Teacher action\n12 dims\n(all four legs)", colors["teacher"])
    _box(ax, (4.2, 1.5), 3.4, 2.1, "Mask out leg-3 obs\nremove 9 dims:\npositions + velocities + last actions", colors["mask"])

    _box(ax, (8.6, 1.7), 3.2, 1.7, "Student input\n39 dims", colors["student"])
    _box(ax, (8.6, 4.95), 3.2, 1.8, "Teacher leg-3 target\n3 dims\n(hip, thigh, calf)", colors["teacher"])

    _box(ax, (12.7, 1.7), 2.2, 1.7, "Student model", colors["student"])
    _box(ax, (12.7, 4.95), 2.2, 1.8, "Student output\n3 dims", colors["output"])

    _arrow(ax, (3.3, 5.85), (4.1, 5.85))
    _arrow(ax, (3.3, 2.7), (4.1, 2.7))
    _arrow(ax, (7.6, 2.7), (8.5, 2.55))
    _arrow(ax, (7.6, 5.85), (8.5, 5.85))
    _arrow(ax, (11.8, 2.55), (12.6, 2.55))
    _arrow(ax, (13.8, 3.4), (13.8, 4.85))

    ax.text(10.2, 7.15, "Training view", fontsize=13, fontweight="bold")
    ax.text(10.15, 0.85, "Inference / control view", fontsize=13, fontweight="bold")
    ax.set_title("Teacher -> Student Interface", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "student_io_diagram.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_supervised_mlp(out_dir)
    draw_student_io(out_dir)
    print(f"Saved interview assets to {out_dir}")


if __name__ == "__main__":
    main()
