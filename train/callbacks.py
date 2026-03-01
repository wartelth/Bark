"""
Experiment tracking callbacks: W&B (native SB3) and Comet (custom).
Use --wandb in train_rl.py for W&B. For Comet, set COMET_API_KEY and use --comet.
"""
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
