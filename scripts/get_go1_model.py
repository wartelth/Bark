"""
Download the Unitree Go1 model from MuJoCo Menagerie so BarkGo1_3Leg-v0 can run.
Run once from repo root:  PYTHONPATH=. python scripts/get_go1_model.py

Creates assets/unitree_go1/scene.xml, go1.xml, and assets/*.stl (meshes).
"""
from pathlib import Path
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go1"
OUT_DIR = REPO_ROOT / "assets" / "unitree_go1"
ASSETS_DIR = OUT_DIR / "assets"

FILES = [
    "scene.xml",
    "go1.xml",
]
MESHES = ("trunk.stl", "hip.stl", "thigh.stl", "thigh_mirror.stl", "calf.stl")


def fetch(url: str, path: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        path = OUT_DIR / name
        print(f"Fetching {name} ...")
        if not fetch(f"{BASE}/{name}", path):
            return 1
        print(f"  -> {path}")

    for name in MESHES:
        path = ASSETS_DIR / name
        print(f"Fetching assets/{name} ...")
        if not fetch(f"{BASE}/assets/{name}", path):
            return 1
        print(f"  -> {path}")

    print("Done. Run the Go1 env with: env_id=BarkGo1_3Leg-v0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
