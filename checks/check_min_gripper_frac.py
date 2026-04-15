"""Sanity check: verify min_gripper_frac by simulating the gripper opening to the
computed angle and checking the fingertip separation against the cube width.

For each cube size in the range, prints:
  - Cube width
  - min_gripper_frac value
  - Actual fingertip separation at that angle
  - Whether it clears the cube (with clearance margin)

Usage:
    python check_min_gripper_frac.py
"""
import numpy as np
from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env
import mujoco

env = LiftCubeProjectionT1V2Env()
env.reset()

gripper_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
lo, hi = env.model.jnt_range[gripper_jid]
addr = env.model.jnt_qposadr[gripper_jid]

print(f"Gripper range: [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]")
from src.envs.lift_cube_projection_t1v2 import _GRIP_CLEARANCE
print(f"Clearance margin: {_GRIP_CLEARANCE * 100:.1f}%\n")
print(f"{'Cube width (mm)':>16} | {'min_gripper_frac':>16} | {'Angle (deg)':>11} | {'Fingertip sep (mm)':>18} | {'Clears?':>7}")
print("-" * 80)

test_half_sizes = np.linspace(*env.cube_size_range, 10)

for hs in test_half_sizes:
    frac = (env._min_gripper_angle_for_cube(hs) - lo) / (hi - lo)
    angle = lo + frac * (hi - lo)

    # Measure actual fingertip separation at this angle
    env.data.qpos[addr] = angle
    mujoco.mj_forward(env.model, env.data)
    sp = env.data.site_xpos[env._static_fingertip_site_id]
    mp = env.data.site_xpos[env._moving_fingertip_site_id]
    sep = float(np.linalg.norm(sp - mp))

    cube_width = hs * 2
    clears = sep >= cube_width
    print(f"{cube_width * 1000:>16.1f} | {frac:>16.3f} | {np.degrees(angle):>11.1f} | {sep * 1000:>18.2f} | {'YES' if clears else 'NO':>7}")

env.close()
print("\nAll values should show 'YES' (sep >= cube_width with clearance margin applied).")
