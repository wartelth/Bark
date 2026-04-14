"""
Single CLI entry point that chains the full post-processing pipeline.

Usage:
    PYTHONPATH=. python -m postpro.run_all
    PYTHONPATH=. python -m postpro.run_all --out reports/my_experiment
    PYTHONPATH=. python -m postpro.run_all --tb-dir logs/tensorboard --rollouts
    PYTHONPATH=. python -m postpro.run_all --run-dir models/prosthetic_rl --type prosthetic_rl

Pipeline stages:
    1. Discover  — find all training artifacts (TB logs, evals, checkpoints)
    2. Metrics   — compute convergence, stability, leg symmetry, policy dynamics
    3. Report    — generate text summary + figures
    4. Compare   — cross-run comparison (if multiple runs found)
    5. Policy    — crack open checkpoints for weight analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from postpro import DEFAULT_TB_DIR, DEFAULT_OUTPUT_DIR, REPO_ROOT
from postpro.load_logs import discover_runs, load_run
from postpro.metrics import compute_derived_metrics
from postpro.report import generate_report
from postpro.compare_runs import compare_all
from postpro.policy_analysis import analyze_all_policies, print_policy_summary


def main():
    parser = argparse.ArgumentParser(
        description="Bark post-processing: learn what happened in training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline (auto-discover everything):
    PYTHONPATH=. python -m postpro.run_all

  Custom output directory:
    PYTHONPATH=. python -m postpro.run_all --out reports/experiment_2

  Specific TB directory:
    PYTHONPATH=. python -m postpro.run_all --tb-dir models/prosthetic_rl/tb_logs

  Single run analysis:
    PYTHONPATH=. python -m postpro.run_all --run-dir models/prosthetic_rl --type prosthetic_rl

  Include action rollouts (slower):
    PYTHONPATH=. python -m postpro.run_all --rollouts

  Skip policy weight analysis:
    PYTHONPATH=. python -m postpro.run_all --no-policy
        """,
    )
    parser.add_argument("--tb-dir", type=str, default=None,
                        help="TensorBoard log directory (default: auto-discover)")
    parser.add_argument("--out", type=str, default=None,
                        help=f"Output directory for reports (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Analyze a single run directory instead of auto-discovery")
    parser.add_argument("--type", type=str, default="rl",
                        choices=["rl", "prosthetic_rl", "teacher", "supervised", "il", "bc"],
                        help="Run type when using --run-dir")
    parser.add_argument("--rollouts", action="store_true",
                        help="Run policy rollouts for action statistics (slower)")
    parser.add_argument("--no-policy", action="store_true",
                        help="Skip policy weight analysis")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip cross-run comparison")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    print("=" * 60)
    print("  BARK Post-Processing Pipeline")
    print("=" * 60)

    # --- Stage 1: Discover ---
    print("\n[1/5] Discovering training artifacts...")

    if args.run_dir:
        run_path = Path(args.run_dir)
        if not run_path.is_absolute():
            run_path = REPO_ROOT / run_path
        runs = [load_run(run_path, args.type)]
        print(f"  Loaded single run: {run_path}")
    else:
        tb_dir = Path(args.tb_dir) if args.tb_dir else DEFAULT_TB_DIR
        if not tb_dir.is_absolute():
            tb_dir = REPO_ROOT / tb_dir
        runs = discover_runs(tb_dir)

    if not runs:
        print("\n  No training artifacts found.")
        print("  Expected locations:")
        print(f"    TB logs:      {DEFAULT_TB_DIR}")
        print(f"    Eval logs:    {REPO_ROOT / 'models' / 'eval_logs'}")
        print(f"    Prosthetic:   {REPO_ROOT / 'models' / 'prosthetic_rl'}")
        print(f"    Teacher:      {REPO_ROOT / 'pretrained' / 'go1_teacher'}")
        print(f"    Supervised:   {REPO_ROOT / 'models' / 'supervised_prosthetic'}")
        print(f"    Imitation:    {REPO_ROOT / 'models' / 'imitation_prosthetic'}")
        print("\n  Run training first, then come back here.")
        sys.exit(1)

    print(f"  Found {len(runs)} run(s):")
    for r in runs:
        n_scalars = len(r.scalars)
        has_eval = "yes" if r.eval_log else "no"
        print(f"    - {r.name} ({r.run_type}): {n_scalars} scalar tags, eval={has_eval}")

    # --- Stage 2: Metrics ---
    print("\n[2/5] Computing derived metrics...")
    metrics = [compute_derived_metrics(r) for r in runs]
    for dm in metrics:
        c = dm.convergence
        if c.total_timesteps > 0:
            print(f"    {dm.run_name}: best={c.best_reward:.2f}, "
                  f"final={c.final_reward:.2f}, "
                  f"crashes={dm.stability.n_crashes}")

    # --- Stage 3: Report ---
    print("\n[3/5] Generating report...")
    generate_report(runs, metrics, out_dir)

    # --- Stage 4: Compare ---
    if not args.no_compare and len(runs) > 1:
        print("\n[4/5] Cross-run comparison...")
        compare_all(runs, metrics, out_dir)
    else:
        print("\n[4/5] Skipping comparison (single run or --no-compare)")

    # --- Stage 5: Policy analysis ---
    if not args.no_policy:
        print("\n[5/5] Policy weight analysis...")
        analyses = analyze_all_policies(out_dir, run_rollouts=args.rollouts)
        if analyses:
            print_policy_summary(analyses)
    else:
        print("\n[5/5] Skipping policy analysis (--no-policy)")

    # --- Done ---
    print("\n" + "=" * 60)
    print(f"  Pipeline complete. Reports saved to: {out_dir}")
    print("=" * 60)
    print(f"\n  Files generated:")

    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<30} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
