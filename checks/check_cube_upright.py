"""Sanity check: verify cube_upright_frac at known tilt angles.

Places the cube at specific rotations and renders it so you can visually confirm
the upright fraction matches what you see.

Usage:
    uv run python check_cube_upright.py
"""
import cv2
import mujoco
import numpy as np
from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env

RENDER_H = 320
RENDER_W = 320

# (label, quaternion [w,x,y,z]) — known tilts
CASES = [
    ("upright 0°",     [1.0, 0.0, 0.0, 0.0]),
    ("tilt 15°",       [np.cos(np.radians(15)/2), np.sin(np.radians(15)/2), 0.0, 0.0]),
    ("tilt 30°",       [np.cos(np.radians(30)/2), np.sin(np.radians(30)/2), 0.0, 0.0]),
    ("tilt 45°",       [np.cos(np.radians(45)/2), np.sin(np.radians(45)/2), 0.0, 0.0]),
    ("tilt 60°",       [np.cos(np.radians(60)/2), np.sin(np.radians(60)/2), 0.0, 0.0]),
    ("on side 90°",    [np.cos(np.radians(90)/2), np.sin(np.radians(90)/2), 0.0, 0.0]),
    ("upside-down",    [0.0, 1.0, 0.0, 0.0]),
]


def main():
    env = LiftCubeProjectionT1V2Env(
        render_mode="rgb_array",
        curriculum_stage=3,
        cam_pos_noise=0.0, cam_euler_noise=0.0,
        obs_noise=0.0, fov_noise=0.0,
    )
    env.reset(seed=0)

    cube_jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_adr = env.model.jnt_qposadr[cube_jnt_id]
    cube_pos = [0.25, 0.0, 0.03]

    renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cube_pos
    cam.distance = 0.25
    cam.azimuth = 135
    cam.elevation = -20

    panels = []
    print(f"{'label':>20}  {'cube_upright':>12}  {'threshold':>10}")
    print("-" * 48)

    for label, quat in CASES:
        env.data.qpos[cube_adr:cube_adr+3] = cube_pos
        env.data.qpos[cube_adr+3:cube_adr+7] = quat
        mujoco.mj_forward(env.model, env.data)

        upright = env._cube_upright_frac()
        passes = upright > 0.85
        print(f"  {label:>18}:  {upright:.3f}   {'PASS' if passes else 'FAIL (tipped)'}")

        renderer.update_scene(env.data, camera=cam)
        img = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
        color = (0, 200, 0) if passes else (0, 0, 220)
        cv2.putText(img, label,           (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"upright={upright:.2f}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(img, "PASS" if passes else "TIPPED", (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        panels.append(img)

    renderer.close()
    env.close()

    grid = np.hstack(panels)
    cv2.imwrite("cube_upright_check.png", grid)
    print(f"\nSaved cube_upright_check.png")


if __name__ == "__main__":
    main()
