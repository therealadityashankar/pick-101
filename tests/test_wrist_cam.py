"""Quick test: render wrist cam frame + segmentation overlay and save to PNG.

Usage:
    python test_wrist_cam.py [--stage 3] [--out wrist_test.png]

No model loading — just resets the env and renders one frame.
"""
import argparse

import cv2
import mujoco
import numpy as np

from src.envs.lift_cube import LiftCubeCartesianEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage (default: 3)")
    parser.add_argument("--closed", action="store_true", help="Close gripper after reset")
    parser.add_argument("--out", type=str, default="wrist_test.png")
    args = parser.parse_args()

    env = LiftCubeCartesianEnv(
        render_mode="rgb_array",
        curriculum_stage=args.stage,
        lock_wrist=True,
    )
    env.reset()

    if args.closed:
        gripper_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        qaddr = env.model.jnt_qposadr[gripper_joint_id]
        lo, hi = env.model.jnt_range[gripper_joint_id]
        env.data.qpos[qaddr] = lo + 0.1 * (hi - lo)  # 90% closed
        mujoco.mj_forward(env.model, env.data)

    # --- Wrist cam RGB frame (640x480 → center-crop 480x480) ---
    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    wrist_renderer.update_scene(env.data, camera="wrist_cam")
    rgb = wrist_renderer.render()  # (480, 640, 3)
    crop_x = (640 - 480) // 2     # 80
    rgb_crop = rgb[:, crop_x: crop_x + 480, :]  # (480, 480, 3)

    # --- Segmentation frame (same crop) ---
    seg_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer.update_scene(env.data, camera="wrist_cam")
    seg_renderer.enable_segmentation_rendering()
    seg = seg_renderer.render()    # (480, 640, 2)  channel 1 = geom id
    seg_renderer.disable_segmentation_rendering()

    # Channel 0 = geom ID, channel 1 = objtype (always mjOBJ_GEOM for rendered geoms)
    geom_id_map = seg[..., 0]
    geom_id_map_crop = geom_id_map[:, crop_x: crop_x + 480]  # (480, 480)

    # Print all unique geom IDs visible to debug
    unique_ids = np.unique(geom_id_map_crop)
    print(f"Unique geom IDs in wrist cam view: {unique_ids}")
    for gid in unique_ids:
        if gid < 0:
            continue
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
        count = int(np.sum(geom_id_map_crop == gid))
        print(f"  geom id {gid:4d}: {name or '(unnamed)':40s}  pixels={count}")

    # Geom IDs we care about
    geom_ids = {
        "static_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"),
        "moving_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"),
        "cube":       mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face"),
    }
    print(f"\nTarget geom IDs: {geom_ids}")

    # --- Draw overlay on a copy ---
    img = rgb_crop.copy()

    for key, gid in geom_ids.items():
        mask = (geom_id_map_crop == gid)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            print(f"  {key}: NOT VISIBLE")
            continue

        if key == "cube":
            # Fit rotated rectangle to get 4 oriented corners
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            corners = cv2.boxPoints(rect).astype(int)  # (4, 2)
            print(f"  cube top-face corners: {corners.tolist()}")
            cv2.drawContours(img, [corners], 0, (0, 220, 0), 2)
            for cx, cy in corners:
                cv2.circle(img, (cx, cy), 4, (0, 255, 100), -1)
            cx0, cy0 = corners[0]
            cv2.putText(img, "cube", (cx0, max(cy0 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        else:
            # Topmost point (min y)
            idx = np.argmin(ys)
            px, py = int(xs[idx]), int(ys[idx])
            color = (0, 80, 255) if key == "static_pad" else (255, 80, 0)
            label = "S" if key == "static_pad" else "M"
            print(f"  {key} topmost: ({px},{py}), total pixels={len(ys)}")
            cv2.circle(img, (px, py), 8, color, -1)
            cv2.putText(img, label, (px + 10, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Side-by-side: raw | annotated
    side_by_side = np.concatenate([rgb_crop, img], axis=1)

    # Save as PNG (convert RGB → BGR for cv2)
    out_path = args.out
    cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"\nSaved to {out_path}  ({side_by_side.shape[1]}x{side_by_side.shape[0]})")

    wrist_renderer.close()
    seg_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
