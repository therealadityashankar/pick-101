"""export_best_run.py — Export or restore the best trained model.

Export copies the best model + vec_normalize into best_run/ so it can be
committed to git and shared without including the full runs/ directory.

Restore copies best_run/ back into runs/ so run_real_t1.py can find it.

Usage:
    # Export current best model to best_run/
    uv run python export_best_run.py export

    # Restore best_run/ back into runs/ (e.g. after fresh clone)
    uv run python export_best_run.py restore

The exported folder is committed to git. The runs/ folder is in .gitignore.
"""

import argparse
import json
import shutil
from pathlib import Path

# ── Source run (the one to export) ───────────────────────────────────────────
SOURCE_RUN  = Path("runs/lift_proj_t1_s3/20260401_145442_resumed")
DEST_DIR    = Path("best_run")

# Files to copy from the run root
ROOT_FILES  = ["vec_normalize.pkl", "vec_normalize_eval.pkl", "config.yaml"]
# Subdirectories to copy entirely
SUBDIRS     = ["best_model"]


def export():
    if not SOURCE_RUN.exists():
        print(f"Source run not found: {SOURCE_RUN}")
        print("Make sure runs/ exists or adjust SOURCE_RUN in this script.")
        return

    if DEST_DIR.exists():
        print(f"Removing existing {DEST_DIR}/...")
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    # Copy root files
    for fname in ROOT_FILES:
        src = SOURCE_RUN / fname
        if src.exists():
            shutil.copy2(src, DEST_DIR / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  Warning: {fname} not found in source run")

    # Copy subdirectories
    for sub in SUBDIRS:
        src = SOURCE_RUN / sub
        if src.exists():
            shutil.copytree(src, DEST_DIR / sub)
            print(f"  Copied {sub}/")
        else:
            print(f"  Warning: {sub}/ not found in source run")

    # Save metadata so restore knows where to put it
    meta = {
        "source_run": str(SOURCE_RUN),
    }
    (DEST_DIR / "export_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nExported to {DEST_DIR}/")
    print("run_real_t1.py already points to this path via T1_RUN.")
    print("Commit best_run/ to git to share the model.")


def restore():
    if not DEST_DIR.exists():
        print(f"{DEST_DIR}/ not found. Nothing to restore.")
        return

    meta_path = DEST_DIR / "export_meta.json"
    if not meta_path.exists():
        print(f"No export_meta.json found in {DEST_DIR}/. Cannot determine restore path.")
        return

    meta = json.loads(meta_path.read_text())
    dest_run = Path(meta["source_run"])
    dest_run.mkdir(parents=True, exist_ok=True)

    print(f"Restoring {DEST_DIR}/ → {dest_run}/")

    for fname in ROOT_FILES:
        src = DEST_DIR / fname
        if src.exists():
            shutil.copy2(src, dest_run / fname)
            print(f"  Restored {fname}")

    for sub in SUBDIRS:
        src = DEST_DIR / sub
        dst = dest_run / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Restored {sub}/")

    print(f"\nRestored to {dest_run}/")
    print("You can now run: uv run python run_real_t1.py --no-robot")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "restore"],
                        help="export: copy best model to best_run/  |  "
                             "restore: copy best_run/ back into runs/")
    args = parser.parse_args()

    if args.command == "export":
        export()
    else:
        restore()


if __name__ == "__main__":
    main()
