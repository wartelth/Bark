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
