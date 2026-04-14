# Supervised vs RL student (same teacher)

Both students use **teacher legs 0–2** and learn **leg 3 only** in `ProstheticGo1-v0` (or hybrid eval in `evaluate/compare.py`).

| | Supervised (`train/train_supervised.py`) | RL (`train/train_prosthetic_rl.py`) |
|---|------------------------------------------|--------------------------------------|
| **Data / signal** | Imitation: `(obs_3leg, action_leg3)` from `data/teacher_rollouts.npz` | On-policy PPO: env reward = mix of forward, tracking vs teacher leg 3, alive |
| **Objective** | Minimize MSE to teacher’s leg‑3 actions | Maximize discounted return (can diverge from teacher if that yields higher return) |
| **Typical behavior** | Lower MSE vs teacher; may be suboptimal for raw task return | Higher return possible; higher MSE vs teacher is possible |

**Effect of a bad teacher:** imitation copies bad leg‑3 labels; RL may escape slightly via reward but still depends on teacher for legs 0–2 and on reward design.

**Effect of a strong teacher:** both can look good; RL often trades off tracking MSE for task performance (as in your `reports/` rollout summaries).
