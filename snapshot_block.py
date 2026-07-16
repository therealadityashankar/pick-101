"""snapshot_block.py — Single-frame block detection → MuJoCo representation.

Grabs ONE camera frame, detects the block ArUco tag, places the block in the
MuJoCo board scene, renders it, and saves a side-by-side PNG. Use this to
quickly iterate on the coordinate mapping without running the full loop.

Usage:
    uv run python snapshot_block.py --camera 0
    uv run python snapshot_block.py --camera 0 --output snap.png
"""

import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import mujoco
import numpy as np

from src.envs.lift_cube import LiftCubeCartesianEnv

# Reuse everything from run_real_ik so the mapping matches the real script.
import run_real_ik as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",       type=int,   default=0)
    parser.add_argument("--block-tag",    type=int,   default=101)
    parser.add_argument("--block-height", type=float, default=15.0)
    parser.add_argument("--output",       type=str,   default="snapshot_block.png")
    args = parser.parse_args()

    scale    = R.OUT_W / R.SQUARE_MM
    inset_mm = R.INSET_MM
    tag_centres_mm = R.board_tag_centres(R.SQUARE_MM, R.TAG_MM, R.GAP_MM, R.N_OUTER, R.N_INNER)

    # ── Grab one camera frame ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return
    for _ in range(5):          # discard warm-up frames
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to grab frame")
        return

    # ── Detect tags ───────────────────────────────────────────────────────────
    detector = aruco.ArucoDetector(R.ARUCO_DICT, R.ARUCO_PARAMS)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det_corners, det_ids, _ = detector.detectMarkers(gray)
    corners_dict = {}
    if det_ids is not None:
        for i, tid in enumerate(det_ids.flatten()):
            corners_dict[int(tid)] = det_corners[i].reshape(4, 2)

    H, n_inliers = R.compute_homography(corners_dict, tag_centres_mm, R.SQUARE_MM)

    block_sim_pos = None
    cx_int = cy_int = 0.0
    if H is not None and args.block_tag in corners_dict:
        img_corners = corners_dict[args.block_tag]
        x_int, y_int = R.get_interior_pos(img_corners, H, scale, inset_mm)
        cx_int, cy_int = R.perspective_correct(x_int, y_int, args.block_height)
        block_sim_pos = R.interior_to_sim(cx_int, cy_int)
        print(f"Block detected  int=({cx_int:.1f},{cy_int:.1f})mm  "
              f"sim=({block_sim_pos[0]:.3f},{block_sim_pos[1]:.3f},{block_sim_pos[2]:.3f})m")
    else:
        print(f"Block tag {args.block_tag} not detected (inliers={n_inliers})")

    # ── Build board scene + place block ───────────────────────────────────────
    scene_xml = R.build_scene_xml(R.SQUARE_MM, R.TAG_MM, R.GAP_MM, R.N_OUTER, R.N_INNER)
    env = LiftCubeCartesianEnv(render_mode="rgb_array", scene_path=scene_xml)
    env.reset()
    R.make_block_jenga(env.model)
    cube_jnt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    adr = env.model.jnt_qposadr[cube_jnt]

    if block_sim_pos is not None:
        env.data.qpos[adr:adr + 3] = block_sim_pos
        env.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        env.data.qvel[adr:adr + 6] = 0
    else:
        env.data.qpos[adr:adr + 3] = [0.0, 0.0, -1.0]   # park below floor
    mujoco.mj_forward(env.model, env.data)

    sim_rgb = env.render(camera="wide")
    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

    # Top-down sim render (arm hidden) — directly comparable to rectified view
    if env._renderer is None:
        env._renderer = mujoco.Renderer(env.model, height=480, width=640)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    env._renderer.update_scene(env.data, camera="topdown", scene_option=opt)
    top_bgr = cv2.cvtColor(env._renderer.render(), cv2.COLOR_RGB2BGR)
    top_bgr = cv2.rotate(top_bgr, cv2.ROTATE_90_CLOCKWISE)
    cv2.putText(top_bgr, "sim topdown", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

    # Rectified top-down camera view with detected/corrected block position
    rect = R.draw_rectified(frame, H, R.SQUARE_MM, R.TAG_MM, corners_dict,
                            tag_centres_mm, args.block_tag, args.block_height,
                            scale, inset_mm)
    cv2.putText(rect, "camera rectified", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

    # ── Annotate camera frame ─────────────────────────────────────────────────
    disp = frame.copy()
    if det_ids is not None:
        for i, tid in enumerate(det_ids.flatten()):
            pts = det_corners[i].reshape(4, 2).astype(np.int32)
            is_block = int(tid) == args.block_tag
            color = (0, 200, 255) if is_block else (0, 255, 0) if int(tid) in tag_centres_mm else (80, 80, 255)
            cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, color, 2)

    # ── Side-by-side ──────────────────────────────────────────────────────────
    h = 480
    def fit(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    combined = np.hstack([fit(disp), fit(sim_bgr), fit(rect), fit(top_bgr)])
    if block_sim_pos:
        cv2.putText(combined, f"int=({cx_int:.1f},{cy_int:.1f})mm  sim=({block_sim_pos[0]:.3f},{block_sim_pos[1]:.3f})m",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    cv2.imwrite(args.output, combined)
    print(f"Saved {args.output}")
    env.close()


if __name__ == "__main__":
    main()
