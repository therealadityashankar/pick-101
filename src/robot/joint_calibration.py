"""Load and apply joint calibration from calibrate_joints_real.py output."""

import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).parent.parent.parent / "calibration/joint_calibration.json"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
REAL_RANGES  = [(-100., 100.)] * 5 + [(0., 100.)]


class JointCalibration:
    """Per-joint linear mapping between real normalized values and sim radians.

    If calibration file is missing or a joint has no fit, falls back to the
    default linear mapping from ctrlrange ↔ REAL_RANGES.
    """

    def __init__(self, path: str | Path | None = None, sim_ranges=None):
        self._cal: dict = {}
        self._sim_ranges = sim_ranges  # shape (6, 2) array from model.actuator_ctrlrange

        path = Path(path) if path else DEFAULT_PATH
        if path.exists():
            with open(path) as f:
                self._cal = json.load(f)
            print(f"[calibration] Loaded from {path}")
            for jname, cal in self._cal.items():
                if cal.get("scale") is not None:
                    print(f"  {jname:15s}  scale={cal['scale']:+.5f}  "
                          f"offset={cal['offset']:+.5f}  R²={cal.get('r2', '?'):.4f}")
        else:
            print(f"[calibration] No file at {path} — using default linear mapping")

    def real_to_sim(self, real_joints: dict, sim_ranges=None) -> np.ndarray:
        """Convert real normalized dict → sim ctrl radians (length 6)."""
        sr = sim_ranges if sim_ranges is not None else self._sim_ranges
        ctrl = np.zeros(6)
        for i, jname in enumerate(JOINT_NAMES):
            val = float(real_joints.get(jname, 0.0))
            cal = self._cal.get(jname, {})
            if cal.get("scale") is not None:
                sim_rad = cal["scale"] * val + cal["offset"]
            else:
                # Default: linear map from REAL_RANGES → ctrlrange
                r_lo, r_hi = REAL_RANGES[i]
                lo, hi = sr[i]
                t = (val - r_lo) / (r_hi - r_lo)
                sim_rad = lo + t * (hi - lo)
            if sr is not None:
                lo, hi = sr[i]
                sim_rad = float(np.clip(sim_rad, lo, hi))
            ctrl[i] = sim_rad
        return ctrl

    def sim_to_real(self, ctrl: np.ndarray, sim_ranges=None) -> dict:
        """Convert sim ctrl radians → real normalized dict."""
        sr = sim_ranges if sim_ranges is not None else self._sim_ranges
        out = {}
        for i, jname in enumerate(JOINT_NAMES):
            sim_rad = float(ctrl[i])
            cal = self._cal.get(jname, {})
            if cal.get("scale") is not None and abs(cal["scale"]) > 1e-9:
                real_val = (sim_rad - cal["offset"]) / cal["scale"]
            else:
                r_lo, r_hi = REAL_RANGES[i]
                lo, hi = sr[i]
                t = (sim_rad - lo) / (hi - lo)
                real_val = r_lo + t * (r_hi - r_lo)
            r_lo, r_hi = REAL_RANGES[i]
            out[jname] = float(np.clip(real_val, r_lo, r_hi))
        return out
