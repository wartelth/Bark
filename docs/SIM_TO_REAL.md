# Sim-to-Real and Jacket Calibration

This document describes how to align simulation with real data from the **jacket** (IMU vest) and how to use domain randomization for better transfer.

## Jacket data and calibration

### Coordinate frame and units

- **IMU axes**: Match your jacket’s mounting (e.g. X forward, Y left, Z up per leg). Document this for each dog/session.
- **Units**: Acceleration in g or m/s²; angular velocity in deg/s or rad/s. Keep the same convention in sim and in the CSV loader.
- **Gravity**: At rest, one accel axis should read ~1g. Use this to check orientation and sign.

### Calibration checklist (per session)

1. **Zero rate**: With the dog still, record gyro for a few seconds; subtract the mean (bias) from all gyro readings.
2. **Scale**: If you have a known motion (e.g. turn 90°), check that integrated gyro matches.
3. **Sync**: If using multiple IMUs, ensure timestamps or sample indices are aligned (same clock or same trigger).

### Using jacket CSV in the pipeline

- **Data loader**: `data/jacket_loader.py` expects columns `IMU1AccelX`, … `IMU4GyroZ` (semicolon- or comma-separated).
- **Reference trajectories**: Run `scripts/jacket_to_reference.py` to convert a CSV to a normalized `.npy` reference for reward shaping or IL.
- **Mapping to sim**: Sim observations (e.g. joint angles/velocities) are not in the same space as IMU. For reward shaping, either:
  - Define a simple mapping (e.g. joint angles → synthetic “IMU-like” signals), or
  - Use the reference only for high-level terms (e.g. step frequency, forward speed) derived from the jacket.

## Domain randomization (sim)

To narrow the sim-to-real gap, randomize the following in simulation so the policy sees varied conditions:

- **Observation noise**: Add Gaussian noise to obs (e.g. scale 0.01–0.05) to mimic IMU/sensor noise.
- **Delay**: Optionally delay the “4th leg” observation by 1–2 steps to mimic real latency.
- **Physics**: If your sim allows, vary mass, friction, and damping within plausible ranges across episodes.
- **Initial state**: Randomize initial joint angles and velocities within a small range so the policy is robust to takeoff conditions.

### Where to add it

- **BarkAnt3LegEnv**: In `_get_obs()`, after `_mask_obs_to_3_legs(obs)`, add something like:
  - `obs = obs + self.np_random.normal(0, self.obs_noise_std, obs.shape)`
- **Config**: Add `obs_noise_std: 0.02` (or 0) to `configs/env_ant_3leg.yaml` and pass it into the env.

## Reference-matching reward (optional)

If you have a reference trajectory from the jacket (e.g. “4th leg” target over time):

- Load the reference in the env (or a wrapper).
- At each step, compare the current 4th-leg state (or its proxy in sim) to the reference at the same phase.
- Add a reward term, e.g. `-||state_4th - ref_4th||` or a normalized similarity, with a small weight so it doesn’t dominate forward/health rewards.

This encourages the policy to produce motions that are close to the real-dog pattern captured by the jacket.
