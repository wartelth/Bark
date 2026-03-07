"""
Experiment tracking callbacks: W&B (native SB3) and Comet (custom).
AMP callback: train discriminator on rollout (s,s') vs expert (s,s').
Use --wandb in train_rl.py for W&B. For Comet, set COMET_API_KEY and use --comet.
"""
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class CometLoggerCallback(BaseCallback):
    """
    Logs SB3 training metrics to Comet ML. Set COMET_API_KEY and pass this callback.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._experiment = None

    def _init_callback(self) -> None:
        try:
            import comet_ml
            self._experiment = comet_ml.Experiment()
        except ImportError:
            if self.verbose:
                print("comet_ml not installed; skipping Comet logging")
            self._experiment = None

    def _on_rollout_end(self) -> None:
        if self._experiment is None or self.logger is None:
            return
        for k, v in self.logger.name_to_value.items():
            if v is not None:
                self._experiment.log_metric(k, v, step=self.num_timesteps)

    def _on_training_end(self) -> None:
        if self._experiment is not None:
            try:
                self._experiment.end()
            except Exception:
                pass


class AMPCallback(BaseCallback):
    """
    After each rollout, trains the AMP discriminator on policy (s,s') vs expert (s,s').
    Expects model to have rollout_buffer with observations of shape (n_steps+1, n_envs, obs_dim).
    """

    def __init__(self, amp_trainer, verbose: int = 0):
        super().__init__(verbose)
        self.amp_trainer = amp_trainer

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        obs = buffer.observations  # (n_steps+1, n_envs, obs_dim) for Box
        if obs is None or obs.size == 0:
            return
        obs = np.asarray(obs)
        # Flatten (n_steps, n_envs, obs_dim) for s and s_next
        s = obs[:-1].reshape(-1, obs.shape[-1])
        s_next = obs[1:].reshape(-1, obs.shape[-1])
        logs = self.amp_trainer.update(s, s_next)
        if self.logger and logs:
            for k, v in logs.items():
                self.logger.record(k, v)


# Ant action space: 8D = leg0(0,1), leg1(2,3), leg2(4,5), leg3(6,7); leg3 is prosthetic/inferred
LEG_ACTION_INDICES_8 = [(0, 2), (2, 4), (4, 6), (6, 8)]
# Go1 action space: 12D = 4 legs × 3 joints (hip, thigh, calf); leg 3 = indices 9–11
LEG_ACTION_INDICES_12 = [(0, 3), (3, 6), (6, 9), (9, 12)]


def _leg_action_indices(n_action: int):
    if n_action == 8:
        return LEG_ACTION_INDICES_8
    if n_action == 12:
        return LEG_ACTION_INDICES_12
    return None


class LegMetricsCallback(BaseCallback):
    """
    Logs per-leg action statistics so you can compare if the prosthetic leg (leg 3)
    trains to behave like the observed legs (0, 1, 2). Records mean and std of
    |action| per leg group to TensorBoard/logger. Supports Ant (8D) and Go1 (12D).
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        if not hasattr(buffer, "actions") or buffer.actions is None:
            return
        actions = np.asarray(buffer.actions)
        if actions.size == 0:
            return
        n_action = actions.shape[-1]
        indices = _leg_action_indices(n_action)
        if indices is None:
            return
        flat = actions.reshape(-1, n_action)
        for leg_idx, (lo, hi) in enumerate(indices):
            leg_acts = flat[:, lo:hi]
            mean_mag = np.abs(leg_acts).mean()
            std_mag = np.abs(leg_acts).std()
            if self.logger:
                self.logger.record(f"leg_{leg_idx}_action_mean_abs", float(mean_mag))
                self.logger.record(f"leg_{leg_idx}_action_std_abs", float(std_mag))
        # Ratio: prosthetic (leg 3) vs average of legs 0,1,2
        leg3_lo, leg3_hi = indices[3]
        other_dim = leg3_lo  # total dims for legs 0,1,2
        leg3_mag = np.abs(flat[:, leg3_lo:leg3_hi]).mean()
        other_mag = np.abs(flat[:, :other_dim]).mean()
        ratio = float(leg3_mag / (other_mag + 1e-8))
        if self.logger:
            self.logger.record("leg3_vs_others_action_ratio", ratio)
