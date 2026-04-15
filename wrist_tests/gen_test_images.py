"""Generate wrist cam test images across varied arm/cube positions.

Usage:
    python wrist_tests/gen_test_images.py [--n 9] [--closed]

Each test saves a separate PNG: test_000.png, test_001.png, ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mujoco
import numpy as np

from src.envs.lift_cube import LiftCubeCartesianEnv

OUT_DIR = Path(__file__).parent


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert Z-axis rotation angle to quaternion [w, x, y, z]."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def get_seg_masks(seg_renderer, data, geom_ids, crop_x):
    seg_renderer.update_scene(data, camera="wrist_cam")
    seg_renderer.enable_segmentation_rendering()
    seg = seg_renderer.render()
    seg_renderer.disable_segmentation_rendering()
    id_map = seg[..., 0][:, crop_x: crop_x + 480]
    return {name: (id_map == gid) for name, gid in geom_ids.items()}


def draw_overlay(rgb_crop, seg_masks):
    img = rgb_crop.copy()

    for key, color, label in [
        ("static_pad", (0, 80, 255), "S"),
        ("moving_pad", (255, 80, 0), "M"),
    ]:
        mask = seg_masks.get(key, np.zeros((1, 1), bool))
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        idx = np.argmin(ys)
        px, py = int(xs[idx]), int(ys[idx])
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.putText(img, label, (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    mask = seg_masks.get("cube", np.zeros((1, 1), bool))
    ys, xs = np.where(mask)
    if len(ys) > 4:
        pts = np.column_stack([xs, ys]).astype(np.float32)
        hull = cv2.convexHull(pts)
        # Approximate to a polygon; epsilon controls how aggressively corners are merged
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(int)
        cv2.polylines(img, [approx], isClosed=True, color=(0, 220, 0), thickness=2)
        for cx, cy in approx:
            cv2.circle(img, (cx, cy), 5, (0, 255, 100), -1)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--closed", action="store_true", help="90%% close gripper")
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    env = LiftCubeCartesianEnv(render_mode="rgb_array", curriculum_stage=3, lock_wrist=True)

    wrist_renderer = mujoco.Renderer(env.model, height=480, width=640)
    seg_renderer = mujoco.Renderer(env.model, height=480, width=640)
    crop_x = (640 - 480) // 2

    geom_ids = {
        "static_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"),
        "moving_pad": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"),
        "cube":       mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_top_face"),
    }

    cube_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_addr = env.model.jnt_qposadr[cube_joint_id]
    gripper_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_qaddr = env.model.jnt_qposadr[gripper_joint_id]
    gripper_lo, gripper_hi = env.model.jnt_range[gripper_joint_id]

    for i in range(args.n):
        env.reset(seed=i * 7)

        # --- Vary cube yaw ---
        cube_yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        env.data.qpos[cube_qpos_addr + 3: cube_qpos_addr + 7] = yaw_to_quat(cube_yaw)

        # --- Vary gripper approach angle via wrist_roll (joint 4) ---
        wrist_roll_offset = rng.uniform(-np.radians(30), np.radians(30))
        env.data.qpos[4] = np.pi / 2 + wrist_roll_offset

        # --- Vary gripper lateral offset (joint 0 = shoulder_pan) ---
        pan_offset = rng.uniform(-np.radians(10), np.radians(10))
        env.data.qpos[0] += pan_offset

        mujoco.mj_forward(env.model, env.data)

        if args.closed:
            env.data.qpos[gripper_qaddr] = gripper_lo + 0.1 * (gripper_hi - gripper_lo)
            mujoco.mj_forward(env.model, env.data)

        # Render
        wrist_renderer.update_scene(env.data, camera="wrist_cam")
        rgb = wrist_renderer.render()[:, crop_x: crop_x + 480, :]
        masks = get_seg_masks(seg_renderer, env.data, geom_ids, crop_x)
        annotated = draw_overlay(rgb, masks)
        pair = np.concatenate([rgb, annotated], axis=1)

        suffix = "_closed" if args.closed else ""
        out = OUT_DIR / f"test_{i:03d}{suffix}.png"
        cv2.imwrite(str(out), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

        found = [k for k, m in masks.items() if m.any()]
        print(f"  [{i+1}/{args.n}] {out.name}  cube_yaw={np.degrees(cube_yaw):.0f}°  detected: {found}")

    wrist_renderer.close()
    seg_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
