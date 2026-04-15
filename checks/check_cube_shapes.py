"""Sanity check: render a grid of random cube widths × heights to verify cuboid randomisation.

Usage:
    uv run python check_cube_shapes.py
"""
import cv2
import mujoco
import numpy as np

from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env

RENDER_H = 300
RENDER_W = 300
COLS = 8
ROWS = 5  # = 40 samples

CAM_LOOKAT   = [0.40, -0.10, 0.06]
CAM_DISTANCE = 0.35
CAM_AZIMUTH  = 135
CAM_ELEVATION = -20


def main():
    env = LiftCubeProjectionT1V2Env(
        render_mode="rgb_array",
        cube_size_range=(0.011, 0.026),
        cube_height_range=(0.010, 0.035),
        cam_pos_noise=0.0,
        cam_euler_noise=0.0,
        obs_noise=0.0,
        fov_noise=0.0,
        lock_wrist=True,
    )

    renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)
    cam = mujoco.MjvCamera()
    cam.lookat[:]  = CAM_LOOKAT
    cam.distance   = CAM_DISTANCE
    cam.azimuth    = CAM_AZIMUTH
    cam.elevation  = CAM_ELEVATION

    static_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "static_fingertip")

    panels = []
    for _ in range(COLS * ROWS):
        env.reset()
        mujoco.mj_forward(env.model, env.data)

        geom_id = env._cube_geom_id
        w = env.model.geom_size[geom_id][0] * 2 * 1000   # full width mm
        h = env.model.geom_size[geom_id][2] * 2 * 1000   # full height mm

        cube_top_z  = env.data.qpos[7] + env.model.geom_size[geom_id][2]  # cube z + half_h
        fingertip_z = env.data.site_xpos[static_site_id][2]
        above = fingertip_z > cube_top_z

        renderer.update_scene(env.data, camera=cam)
        img = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)

        color = (0, 220, 0) if above else (0, 0, 220)
        cv2.putText(img, f"w={w:.0f}mm  h={h:.0f}mm",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"tip={'ABOVE' if above else 'BELOW'} top",
                    (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        panels.append(img)

    renderer.close()
    env.close()

    n_rows = (len(panels) + COLS - 1) // COLS
    blank = np.full((RENDER_H, RENDER_W, 3), 30, dtype=np.uint8)
    while len(panels) < n_rows * COLS:
        panels.append(blank)
    grid = np.vstack([np.hstack(panels[r * COLS:(r + 1) * COLS]) for r in range(n_rows)])
    cv2.imwrite("cube_shapes_check.png", grid)
    print(f"Saved cube_shapes_check.png  ({grid.shape[1]}x{grid.shape[0]}px)")


if __name__ == "__main__":
    main()
