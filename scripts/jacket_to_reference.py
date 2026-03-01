"""
Convert jacket CSV to reference trajectory .npy for reward shaping or IL.
Usage:
  PYTHONPATH=. python -m scripts.jacket_to_reference data/raw/jacket_walk.csv -o data/refs/walk_ref.npy
"""
import argparse
from pathlib import Path

from data.reference_trajectory import jacket_to_reference


def main():
    parser = argparse.ArgumentParser(description="Convert jacket CSV to reference .npy")
    parser.add_argument("csv", type=str, help="Path to jacket CSV")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output .npy path")
    parser.add_argument("--sep", type=str, default=";", help="CSV separator")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization")
    parser.add_argument("--max-steps", type=int, default=None, help="Max timesteps to use")
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = Path(args.csv).with_suffix(".npy")
        out = Path("data/refs") / out.name
    ref = jacket_to_reference(
        args.csv,
        out_path=out,
        sep=args.sep,
        normalize=not args.no_normalize,
        max_steps=args.max_steps,
    )
    print(f"Saved reference shape {ref.shape} to {out}")


if __name__ == "__main__":
    main()
