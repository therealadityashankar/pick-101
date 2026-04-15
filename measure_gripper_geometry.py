"""Measure gripper fingertip separation distance as a function of joint angle.

Loads the MuJoCo model, sweeps the gripper joint from min to max,
records the Euclidean distance between static_fingertip and moving_fingertip
sites, and fits a trigonometric model.

Usage:
    python measure_gripper_geometry.py
    python measure_gripper_geometry.py --plot
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = Path("models/so101/lift_cube.xml")
N_SAMPLES = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot distance vs angle")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")
    moving_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "moving_fingertip")
    gripper_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    lo, hi = model.jnt_range[gripper_jid]
    addr = model.jnt_qposadr[gripper_jid]

    print(f"Gripper joint range: [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]")
    print(f"Sweeping {N_SAMPLES} samples...\n")

    angles = np.linspace(lo, hi, N_SAMPLES)
    distances = np.zeros(N_SAMPLES)

    for i, angle in enumerate(angles):
        data.qpos[addr] = angle
        mujoco.mj_forward(model, data)
        sp = data.site_xpos[static_id]
        mp = data.site_xpos[moving_id]
        distances[i] = float(np.linalg.norm(sp - mp))

    print("Angle (deg) | Distance (mm)")
    print("-" * 30)
    for angle_deg, dist_m in zip(
        np.degrees(angles[::N_SAMPLES // 20]),
        distances[::N_SAMPLES // 20],
    ):
        print(f"  {angle_deg:7.1f}   |  {dist_m * 1000:7.2f}")

    # For grabbing a 3cm cube (0.015 half-size), need distance > 0.030 + margin
    cube_widths = np.array([0.020, 0.024, 0.028, 0.030, 0.034, 0.036, 0.040])
    print("\nMin gripper angle to clear cube:")
    print("Cube width (mm) | Min angle (deg) | Gripper fraction")
    print("-" * 55)
    for cw in cube_widths:
        gap = cw * 1.15  # 15% clearance margin
        if gap <= distances.min():
            angle_needed = lo
        elif gap >= distances.max():
            angle_needed = hi
        else:
            angle_needed = float(np.interp(gap, distances, angles))
        frac = (angle_needed - lo) / (hi - lo)
        print(f"  {cw * 1000:6.1f}         |  {np.degrees(angle_needed):8.1f}      |  {frac:.3f}")

    # Quick linear fit for simple reference
    coeffs = np.polyfit(angles, distances, 1)
    print(f"\nLinear fit: distance ≈ {coeffs[0]*1000:.2f}*angle_rad + {coeffs[1]*1000:.2f}  (mm)")
    print(f"Distance range: [{distances.min()*1000:.1f}, {distances.max()*1000:.1f}] mm")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(np.degrees(angles), distances * 1000, "b-", linewidth=2)
            plt.xlabel("Gripper joint angle (degrees)")
            plt.ylabel("Fingertip separation distance (mm)")
            plt.title("SO-101 Gripper: Fingertip Distance vs Joint Angle")
            plt.grid(True, alpha=0.4)
            plt.axhline(30, color="r", linestyle="--", label="3cm cube width")
            plt.legend()
            plt.tight_layout()
            plt.savefig("gripper_geometry.png", dpi=120)
            print("\nPlot saved to gripper_geometry.png")
            plt.show()
        except ImportError:
            print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
