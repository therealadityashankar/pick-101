"""Test: verify 3D projection-based cube top corners against color-segmented ground truth.

Ground truth: render wrist cam, isolate red cube pixels via color threshold, find the
topmost convex hull (top face visible from above).

Projection: use LiftCubeProjectionT1V2Env._get_cube_top_corners_world() + camera matrices.

Usage:
    uv run python test_cube_corner_projection.py --trials 10
"""
import argparse
import os

import cv2
import mujoco
import numpy as np

from src.envs.lift_cube_projection_t1v2 import LiftCubeProjectionT1V2Env

# Red cube color bounds in RGB
CUBE_R_LOW  = np.array([150,  10,  10], dtype=np.uint8)
CUBE_R_HIGH = np.array([255, 120, 120], dtype=np.uint8)


def get_color_corners_px(frame_rgb: np.ndarray):
    """Find top-face corners of red cube via color mask + convex hull."""
    mask = cv2.inRange(frame_rgb, CUBE_R_LOW, CUBE_R_HIGH)
    # Only keep top 70% of image — top face is higher in frame than sides
    h = mask.shape[0]
    mask[int(h * 0.7):] = 0
    ys, xs = np.where(mask > 0)
    if len(ys) < 10:
        return None, mask
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
    return approx.astype(np.float32), mask


def draw_corners(img, corners_px, color, label_prefix):
    out = img.copy()
    ci = corners_px.astype(int)
    cv2.polylines(out, [ci], isClosed=True, color=color, thickness=2)
    for i, (x, y) in enumerate(ci):
        cv2.circle(out, (x, y), 5, color, -1)
        cv2.putText(out, f"{label_prefix}{i}", (x + 6, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",      type=int, default=10)
    parser.add_argument("--stage",       type=int, default=3)
    parser.add_argument("--output",      type=str, default="corner_comparison.png")
    parser.add_argument("--visuals-dir", type=str, default="projection_test_visuals")
    args = parser.parse_args()

    os.makedirs(args.visuals_dir, exist_ok=True)

    env = LiftCubeProjectionT1V2Env(
        render_mode="rgb_array",
        curriculum_stage=args.stage,
        cam_pos_noise=0.0,
        cam_euler_noise=0.0,
        obs_noise=0.0,
        fov_noise=0.0,
        cube_size_range=(0.013, 0.026),
        cube_height_range=(0.010, 0.035),
    )
    crop_x = (env.CAM_W - env.CAM_H) // 2
    renderer = mujoco.Renderer(env.model, height=env.CAM_H, width=env.CAM_W)

    all_errors = []

    for trial in range(args.trials):
        env.reset(seed=trial)
        mujoco.mj_forward(env.model, env.data)

        gsize  = env.model.geom_size[env._cube_geom_id]
        w_mm   = gsize[0] * 2000
        h_mm   = gsize[2] * 2000

        # Render wrist cam and square-crop
        renderer.update_scene(env.data, camera="wrist_cam")
        frame_full = renderer.render().copy()
        frame = frame_full[:, crop_x:crop_x + env.CAM_H, :]

        # Ground truth: color-based corners
        color_corners, mask = get_color_corners_px(frame)

        # Projected corners from env
        K, R, t = env._get_camera_matrices()
        world_corners = env._get_cube_top_corners_world()
        proj_norm = env._project_points_norm(world_corners, K, R, t)
        proj_px = proj_norm * np.array([env.CAM_H, env.CAM_H])

        # Error: nearest-neighbour match
        if color_corners is not None and len(color_corners) >= 3:
            errors = [float(np.min(np.linalg.norm(proj_px - sp, axis=1))) for sp in color_corners]
            mean_err = float(np.mean(errors))
            all_errors.append(mean_err)
            err_str = f"mean={mean_err:.1f}px  [{' '.join(f'{e:.0f}' for e in errors)}]"
        else:
            err_str = "color corners not found (cube may not be visible)"
            color_corners = np.array([[0, 0]], dtype=np.float32)

        print(f"Trial {trial+1:2d}: w={w_mm:.0f}mm h={h_mm:.0f}mm  {err_str}")

        # Build comparison image
        img_proj  = draw_corners(frame.copy(), proj_px,       (100, 100, 255), "P")
        img_color = draw_corners(frame.copy(), color_corners, (0,   220,   0), "C")
        img_both  = draw_corners(img_proj,     color_corners, (0,   220,   0), "C")

        for img, lbl in [
            (img_proj,  f"Projection w={w_mm:.0f} h={h_mm:.0f}mm"),
            (img_color, "Color GT (green)"),
            (img_both,  err_str[:45]),
        ]:
            cv2.putText(img, lbl, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.concatenate([
            cv2.cvtColor(img_proj,  cv2.COLOR_RGB2BGR),
            cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(img_both,  cv2.COLOR_RGB2BGR),
        ], axis=1)
        cv2.imwrite(os.path.join(args.visuals_dir, f"trial_{trial+1:04d}.png"), combined)
        if trial == args.trials - 1:
            cv2.imwrite(args.output, combined)
            print(f"Saved final comparison to {args.output}")

    renderer.close()
    env.close()

    if all_errors:
        print(f"\n{'='*55}")
        print(f"Summary across {len(all_errors)} trials:")
        print(f"  Mean : {np.mean(all_errors):.1f}px")
        print(f"  Std  : {np.std(all_errors):.1f}px")
        print(f"  Min  : {np.min(all_errors):.1f}px")
        print(f"  Max  : {np.max(all_errors):.1f}px")
        print(f"  Visuals: {os.path.abspath(args.visuals_dir)}/")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
