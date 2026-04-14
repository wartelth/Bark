"""
Bark post-processing pipeline.

Unified analysis of everything that happened during training:
TensorBoard logs, evaluation checkpoints, supervised losses,
policy weights, and cross-run comparisons.

Usage (full pipeline):
    PYTHONPATH=. python -m postpro.run_all
    PYTHONPATH=. python -m postpro.run_all --runs-dir logs/tensorboard --out reports/

Usage (individual modules):
    from postpro.load_logs import discover_runs, load_run
    from postpro.metrics import compute_derived_metrics
    from postpro.report import generate_report
    from postpro.compare_runs import compare_all
    from postpro.policy_analysis import analyze_policy
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TB_DIR = REPO_ROOT / "logs" / "tensorboard"
DEFAULT_EVAL_DIR = REPO_ROOT / "models" / "eval_logs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports"

PROSTHETIC_RL_DIR = REPO_ROOT / "models" / "prosthetic_rl"
SUPERVISED_DIR = REPO_ROOT / "models" / "supervised_prosthetic"
IMITATION_DIR = REPO_ROOT / "models" / "imitation_prosthetic"
TEACHER_DIR = REPO_ROOT / "pretrained" / "go1_teacher"

ALL_MODEL_DIRS = {
    "rl_main": REPO_ROOT / "models",
    "prosthetic_rl": PROSTHETIC_RL_DIR,
    "supervised": SUPERVISED_DIR,
    "imitation": IMITATION_DIR,
    "teacher": TEACHER_DIR,
}
