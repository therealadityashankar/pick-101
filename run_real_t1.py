"""run_real_t1.py — Run T1 policy on the real robot, guided by IRL cube detection.

Architecture:
  1. Overhead camera detects cube ArUco tag → board mm position
  2. Perspective correction → true interior position
  3. Sim cube is placed at the detected IRL position (last-known if lost)
  4. Real robot joint positions are mirrored into the sim each step
  5. T1 policy observes the sim state (wrist-cam projections of fingertips + cube)
  6. Policy outputs a Cartesian action → sim IK step
  7. Resulting sim joint targets are sent to the real robot
  8. 4-panel + debug bar video is recorded

Usage:
    uv run python run_real_t1.py --port /dev/tty.usbmodem585A0076891
    uv run python run_real_t1.py --no-robot --camera 0
"""

import argparse
import atexit
import tempfile
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube import LiftCubeCartesianEnv
from src.envs.lift_cube_projection import LiftCubeProjectionEnv
from src.robot.joint_calibration import JointCalibration
from eval_projection import draw_obs_from_array

# ── Policy path ───────────────────────────────────────────────────────────────
T1_RUN     = Path("runs/lift_proj_t1_s3/20260401_145442_resumed")
T1_MODEL   = T1_RUN / "best_model/best_model.zip"
T1_VECNORM = T1_RUN / "vec_normalize.pkl"

# ── ArUco detection (must match visualize_irl_block.py) ───────────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.errorCorrectionRate       = 0.7

BORDERED_IDS = set(range(100, 200))
MARGIN_FRAC  = 48 / 200

SQUARE_MM = 180.0
TAG_MM    = 16.0
GAP_MM    = 2.0
N_OUTER   = 10
N_INNER   = 8
INSET_MM  = 2 * TAG_MM + GAP_MM   # 34mm

OUT_W = 600
OUT_H = 600

# ── Manual position offset (tune these to correct block position) ─────────────
DELTA_X =  -5.0   # mm to add to observed X
DELTA_Y =   3   # mm to add to observed Y

# ── Coordinate mapping ────────────────────────────────────────────────────────
ROBOT_INT_X      = 56.0    # interior mm — lateral centre of robot base
ROBOT_INT_Y      = 146.0   # interior mm — back edge of board (where robot sits)
ROBOT_BASE_TIP_X = 0.055   # sim metres — front face of robot base from world origin
CUBE_HALF        = 0.015   # sim metres — half the cube side
CUBE_HALF_MM     = CUBE_HALF * 1000

# ── Joint calibration (loaded from calibration/joint_calibration.json if present) ──
# Run calibrate_joints_real.py first to generate the calibration file.
# Falls back to a default linear mapping if no file exists.
_calibration: JointCalibration | None = None  # initialised in main()


def get_calibration(sim_ranges) -> JointCalibration:
    global _calibration
    if _calibration is None:
        _calibration = JointCalibration(sim_ranges=sim_ranges)
    return _calibration


# ─────────────────────────────────────────────────────────────────────────────
# Board detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    centres = {}
    tag_id  = 0

    def side_centres(inset_mm, n, side):
        half   = tag_mm / 2
        margin = inset_mm + half
        span   = square_mm - 2 * margin
        positions = []
        for i in range(n):
            t     = i / (n - 1) if n > 1 else 0.5
            along = margin + t * span
            if side == "top":
                positions.append((along, inset_mm + half))
            elif side == "bottom":
                positions.append((along, square_mm - inset_mm - half))
            elif side == "left":
                if 0 < i < n - 1:
                    positions.append((inset_mm + half, along))
            elif side == "right":
                if 0 < i < n - 1:
                    positions.append((square_mm - inset_mm - half, along))
        return positions

    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(0, n_outer, side):
            centres[tag_id] = (cx, cy)
            tag_id += 1
    inner_inset = tag_mm + gap_mm
    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(inner_inset, n_inner, side):
            centres[tag_id] = (cx, cy)
            tag_id += 1
    return centres


def compute_homography(corners_dict, tag_centres_mm, square_mm):
    src_pts, dst_pts = [], []
    for tag_id, img_corners in corners_dict.items():
        if tag_id not in tag_centres_mm:
            continue
        src_pts.append([float(img_corners[:, 0].mean()), float(img_corners[:, 1].mean())])
        cx_mm, cy_mm = tag_centres_mm[tag_id]
        dst_pts.append([cx_mm * OUT_W / square_mm, cy_mm * OUT_W / square_mm])
    if len(src_pts) < 4:
        return None, len(src_pts)
    H, mask = cv2.findHomography(
        np.array(src_pts, np.float32), np.array(dst_pts, np.float32), cv2.RANSAC, 5.0)
    return H, int(mask.sum()) if mask is not None else 0


def bordered_corners_image(img_corners):
    tl, tr, br, bl = img_corners.astype(np.float64)
    right = tr - tl; down = bl - tl; m = MARGIN_FRAC
    return np.array([tl - m*right - m*down, tr + m*right - m*down,
                     br + m*right + m*down, bl - m*right + m*down], dtype=np.float64)


def get_interior_pos(img_corners, H, scale, inset_mm):
    # Use the TL corner of the ArUco tag
    tl_img = img_corners[0].astype(np.float32)
    board_pt = cv2.perspectiveTransform(tl_img.reshape(1, 1, 2), H).reshape(2)
    return float(board_pt[0]) / scale - inset_mm, float(board_pt[1]) / scale - inset_mm


def perspective_correct(obs_x, obs_y, h_mm=None):
    return obs_x + DELTA_X, obs_y + DELTA_Y


def interior_to_sim(int_x_mm, int_y_mm):
    """Interior mm → sim cube-centre (x, y, z).

    Reference point is the ArUco tag TL corner.
    Offset by CUBE_HALF_MM to shift from TL corner to cube centre.
    """
    sim_x = ROBOT_BASE_TIP_X + (ROBOT_INT_Y - (int_y_mm + CUBE_HALF_MM)) / 1000.0
    sim_y = (ROBOT_INT_X - (int_x_mm + CUBE_HALF_MM)) / 1000.0
    return sim_x, sim_y, CUBE_HALF


# ─────────────────────────────────────────────────────────────────────────────
# Rectified top-down view (panel 3)
# ─────────────────────────────────────────────────────────────────────────────

def draw_rectified(frame, H, square_mm, tag_mm, corners_dict,
                   tag_centres_mm, block_tag, block_height, scale, inset_mm):
    if H is None:
        blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No homography", (20, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return blank

    warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))

    interior_size_mm = square_mm - 2 * inset_mm
    inset_px = int(inset_mm * scale)
    isize_px = int(interior_size_mm * scale)

    cv2.rectangle(warped, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)
    cv2.rectangle(warped, (inset_px, inset_px),
                  (inset_px + isize_px, inset_px + isize_px), (0, 0, 0), 2)

    for d_mm in range(10, int(interior_size_mm), 10):
        d_px  = inset_px + int(d_mm * scale)
        color = (80, 80, 80) if d_mm % 20 == 0 else (160, 160, 160)
        cv2.line(warped, (d_px, inset_px), (d_px, inset_px + isize_px), color, 1)
        cv2.line(warped, (inset_px, d_px), (inset_px + isize_px, d_px), color, 1)

    for tid, (cx_mm, cy_mm) in tag_centres_mm.items():
        cv2.circle(warped, (int(cx_mm * scale), int(cy_mm * scale)), 3, (0, 160, 0), -1)

    if block_tag in corners_dict:
        img_corners = corners_dict[block_tag]
        # Draw bordered tag outline
        b_img   = bordered_corners_image(img_corners)
        b_board = cv2.perspectiveTransform(
            b_img.astype(np.float32).reshape(1, -1, 2), H).reshape(-1, 2)
        cv2.polylines(warped, [b_board.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 2)
        # Raw ref point: ArUco TL corner in board space (same as get_interior_pos)
        tl_board = cv2.perspectiveTransform(
            img_corners[0].astype(np.float32).reshape(1, 1, 2), H).reshape(2)
        tl_px = (int(tl_board[0]), int(tl_board[1]))
        x_int = float(tl_board[0]) / scale - inset_mm
        y_int = float(tl_board[1]) / scale - inset_mm
        cv2.drawMarker(warped, tl_px, (0, 200, 255), cv2.MARKER_CROSS, 14, 2)
        # Corrected position (what interior_to_sim actually receives)
        cx_int, cy_int = perspective_correct(x_int, y_int, block_height)
        cx_px = inset_px + int(cx_int * scale)
        cy_px = inset_px + int(cy_int * scale)
        cv2.circle(warped, (cx_px, cy_px), 7, (0, 255, 0), -1)
        cv2.putText(warped, f"({cx_int:.1f},{cy_int:.1f})mm",
                    (cx_px + 8, cy_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return warped


# ─────────────────────────────────────────────────────────────────────────────
# Board scene builder (for visualization env)
# ─────────────────────────────────────────────────────────────────────────────

def board_to_sim(board_x_mm, board_y_mm):
    sim_x = ROBOT_BASE_TIP_X + (SQUARE_MM - board_y_mm) / 1000.0
    sim_y = -(board_x_mm - SQUARE_MM / 2) / 1000.0
    return sim_x, sim_y


def build_scene_xml(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    tag_half  = tag_mm / 2 / 1000
    board_h   = square_mm / 2 / 1000
    inset_mm  = 2 * tag_mm + gap_mm
    int_half  = (square_mm / 2 - inset_mm) / 1000
    board_cx, board_cy = board_to_sim(square_mm / 2, square_mm / 2)

    geoms = []
    geoms.append(
        f'<geom name="board_surface" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} -0.0004" '
        f'size="{board_h:.4f} {board_h:.4f} 0.0004" '
        f'rgba="0.85 0.80 0.65 1" contype="0" conaffinity="0" group="1"/>'
    )
    geoms.append(
        f'<geom name="board_interior" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} 0.00005" '
        f'size="{int_half:.4f} {int_half:.4f} 0.00005" '
        f'rgba="0.92 0.88 0.72 1" contype="0" conaffinity="0" group="1"/>'
    )
    tag_centres = board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner)
    for i, (bx, by) in tag_centres.items():
        sx, sy = board_to_sim(bx, by)
        geoms.append(
            f'<geom name="btag_{i}" type="box" '
            f'pos="{sx:.4f} {sy:.4f} 0.0001" '
            f'size="{tag_half:.4f} {tag_half:.4f} 0.0001" '
            f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
        )
    geoms.append(
        f'<geom name="base_tip_line" type="box" '
        f'pos="{ROBOT_BASE_TIP_X:.4f} 0.0000 0.0003" '
        f'size="0.0008 {board_h:.4f} 0.0003" '
        f'rgba="1 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
    )
    geoms_xml = "\n        ".join(geoms)
    topdown_cam = (
        f'<camera name="topdown" pos="{board_cx:.4f} {board_cy:.4f} 0.6" '
        f'euler="0 0 3.14159" fovy="30"/>'
    )
    base_xml = Path(__file__).parent / "models/so101/lift_cube.xml"
    xml = f"""<mujoco model="lift_cube_board">
    <include file="{base_xml}"/>
    <worldbody>
        {geoms_xml}
        {topdown_cam}
    </worldbody>
</mujoco>"""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False,
        dir=str(Path(__file__).parent / "models/so101")
    )
    tmp.write(xml.encode())
    tmp.close()
    atexit.register(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",         type=str,   default=None,
                        help="Serial port for SO-101 (e.g. /dev/tty.usbmodem...)")
    parser.add_argument("--camera",       type=int,   default=0)
    parser.add_argument("--block-tag",    type=int,   default=101)
    parser.add_argument("--block-height", type=float, default=15.0,
                        help="Height of tag above board in mm (cube top = 30mm)")
    parser.add_argument("--no-robot",     action="store_true",
                        help="Run without real robot (sim only, for testing)")
    parser.add_argument("--output",       type=str,   default="real_t1_run.mp4")
    parser.add_argument("--fps",          type=float, default=10.0)
    parser.add_argument("--max-steps",    type=int,   default=350)
    args = parser.parse_args()

    if args.port is None and not args.no_robot:
        parser.error("Provide --port or use --no-robot")

    scale    = OUT_W / SQUARE_MM
    inset_mm = INSET_MM
    tag_centres_mm = board_tag_centres(SQUARE_MM, TAG_MM, GAP_MM, N_OUTER, N_INNER)

    # ── Policy env (13-dim obs, standard simplified XML) ─────────────────────
    print("Loading policy env...")
    policy_env = LiftCubeProjectionEnv(
        render_mode=None,
        curriculum_stage=3,
        lock_wrist=True,
        cam_pos_noise=0.0,
        cam_euler_noise=0.0,
        obs_noise=0.0,
        reward_version="v_straddle",
    )
    vec_env = DummyVecEnv([lambda: policy_env])
    vec_env = VecNormalize.load(str(T1_VECNORM), vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    print("Loading policy weights...")
    policy = PPO.load(str(T1_MODEL))

    sim_ranges = policy_env.model.actuator_ctrlrange[:6].copy()
    cal        = get_calibration(sim_ranges)

    # Cube joint in policy env
    cube_jnt_id   = mujoco.mj_name2id(policy_env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_adr = policy_env.model.jnt_qposadr[cube_jnt_id]

    # ── Visualization env (board scene, full XML) ─────────────────────────────
    print("Building board scene...")
    scene_xml = build_scene_xml(SQUARE_MM, TAG_MM, GAP_MM, N_OUTER, N_INNER)
    viz_env = LiftCubeCartesianEnv(render_mode="rgb_array", scene_path=scene_xml)
    viz_env.reset()

    viz_cube_jnt_id   = mujoco.mj_name2id(viz_env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    viz_cube_qpos_adr = viz_env.model.jnt_qposadr[viz_cube_jnt_id]

    # ── Connect to real robot ─────────────────────────────────────────────────
    robot = None
    if not args.no_robot:
        from src.robot.real_robot import RealRobot
        robot = RealRobot(port=args.port)
        robot.connect()
        robot.enable_torque()
        print("Connected to real robot")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    # ── Reset policy env ──────────────────────────────────────────────────────
    obs = vec_env.reset()
    if robot is not None:
        real_joints  = robot.get_joint_positions()
        init_ctrl    = cal.real_to_sim(real_joints, sim_ranges)
        policy_env.data.ctrl[:6] = init_ctrl
        policy_env.data.qpos[:6] = init_ctrl
        mujoco.mj_forward(policy_env.model, policy_env.data)
        raw_obs = policy_env._get_obs()
        obs = vec_env.normalize_obs(raw_obs[np.newaxis])

    # Park cube below floor until first detection
    for env_ref, qadr in [(policy_env, cube_qpos_adr), (viz_env, viz_cube_qpos_adr)]:
        env_ref.data.qpos[qadr:qadr + 3] = [0.0, 0.0, -1.0]
        env_ref.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(env_ref.model, env_ref.data)

    # Ensure viz renderer is initialised for topdown rendering
    if viz_env._renderer is None:
        viz_env._renderer = mujoco.Renderer(viz_env.model, height=480, width=640)

    # Wrist renderer for panel 5 (uses policy_env model — same camera as policy obs)
    wrist_renderer = mujoco.Renderer(policy_env.model, height=480, width=640)
    _crop_x = policy_env._crop_x

    # ── Recording ─────────────────────────────────────────────────────────────
    writer       = None
    frame_count  = 0
    block_sim_pos = None
    H = None
    n_inliers = 0
    cx_int = cy_int = 0.0
    detected_last = False

    print(f"Running T1 for up to {args.max_steps} steps. Recording → {args.output}")

    try:
        for step in range(args.max_steps):
            t_start = time.time()

            # ── Detect cube ───────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det_corners, det_ids, _ = detector.detectMarkers(gray)
            corners_dict = {}
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    corners_dict[int(tid)] = det_corners[i].reshape(4, 2)

            H, n_inliers = compute_homography(corners_dict, tag_centres_mm, SQUARE_MM)

            if H is not None and args.block_tag in corners_dict:
                img_corners = corners_dict[args.block_tag]
                x_int, y_int = get_interior_pos(img_corners, H, scale, inset_mm)
                cx_int, cy_int = perspective_correct(x_int, y_int, args.block_height)
                block_sim_pos = interior_to_sim(cx_int, cy_int)
                if not detected_last:
                    print(f"[{step:4d}] Block detected  "
                          f"int=({cx_int:.1f},{cy_int:.1f})mm  "
                          f"sim=({block_sim_pos[0]:.3f},{block_sim_pos[1]:.3f})m")
                detected_last = True
            else:
                if detected_last:
                    print(f"[{step:4d}] Block lost — using last known position")
                detected_last = False

            # ── Sync cube in both envs ─────────────────────────────────────────
            if block_sim_pos is not None:
                for env_ref, qadr in [(policy_env, cube_qpos_adr), (viz_env, viz_cube_qpos_adr)]:
                    env_ref.data.qpos[qadr:qadr + 3] = block_sim_pos
                    env_ref.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
                    env_ref.data.qvel[qadr:qadr + 6] = 0

            # ── Sync arm from real robot ───────────────────────────────────────
            if robot is not None:
                real_joints = robot.get_joint_positions()
                sim_ctrl    = cal.real_to_sim(real_joints, sim_ranges)
                if step % 10 == 0:
                    print(f"  pan: real={real_joints.get('shoulder_pan', '?'):+.1f}  "
                          f"sim={sim_ctrl[0]:+.3f}rad")
                policy_env.data.ctrl[:6] = sim_ctrl
                policy_env.data.qpos[:6] = sim_ctrl
                mujoco.mj_forward(policy_env.model, policy_env.data)

            # ── Run policy ────────────────────────────────────────────────────
            raw_obs = policy_env._get_obs()
            norm_obs = vec_env.normalize_obs(raw_obs[np.newaxis])
            action, _ = policy.predict(norm_obs, deterministic=True)

            # ── Step policy env ───────────────────────────────────────────────
            obs, reward, done, info = vec_env.step(action)

            # ── Send targets to real robot ─────────────────────────────────────
            if robot is not None:
                target_ctrl  = policy_env.data.ctrl[:6].copy()
                real_targets = cal.sim_to_real(target_ctrl, sim_ranges)
                robot.set_joint_positions(real_targets)

            # ── Mirror arm state from policy env → viz env ────────────────────
            viz_env.data.ctrl[:] = policy_env.data.ctrl[:]
            viz_env.data.qpos[:7] = policy_env.data.qpos[:7]  # arm joints
            viz_env.data.qvel[:6] = policy_env.data.qvel[:6]
            mujoco.mj_forward(viz_env.model, viz_env.data)

            # ── Panel 1: annotated camera frame ───────────────────────────────
            disp = frame.copy()
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    pts = det_corners[i].reshape(4, 2).astype(np.int32)
                    is_border = int(tid) in tag_centres_mm
                    is_block  = int(tid) == args.block_tag
                    color = (0, 255, 0) if is_border else (0, 200, 255) if is_block else (80, 80, 255)
                    cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, color, 2)
                    cx_px2, cy_px2 = pts.mean(axis=0).astype(int)
                    cv2.putText(disp, str(tid), (cx_px2 - 8, cy_px2 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(disp, "Camera", (8, disp.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 2: simulation side view (via viz env) ───────────────────
            sim_rgb = viz_env.render(camera="wide")
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(sim_bgr, f"T1 policy  reward={float(reward[0]):.3f}",
                        (8, sim_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 3: rectified top-down board view ────────────────────────
            rect = draw_rectified(frame, H, SQUARE_MM, TAG_MM, corners_dict,
                                  tag_centres_mm, args.block_tag, args.block_height,
                                  scale, inset_mm)
            cv2.putText(rect, "Top-down (homography)", (8, rect.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 4: top-down sim camera, arm hidden ──────────────────────
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[0] = 1
            opt.geomgroup[1] = 1
            viz_env._renderer.update_scene(viz_env.data, camera="topdown", scene_option=opt)
            top_rgb = viz_env._renderer.render()
            top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
            top_bgr = cv2.rotate(top_bgr, cv2.ROTATE_90_CLOCKWISE)
            cv2.putText(top_bgr, "Simulation (top-down)", (8, top_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 5: wrist camera with obs keypoints ──────────────────────
            wrist_renderer.update_scene(policy_env.data, camera="wrist_cam")
            wrist_raw = wrist_renderer.render()[:, _crop_x:_crop_x + 480, :]   # 480×480 RGB
            raw_obs = policy_env._get_obs()
            extra_labels = [
                ("reward", f"{float(reward[0]):+.3f}"),
                ("cube_z", f"{info[0].get('cube_z', 0.0):.3f}"),
                ("inliers", str(n_inliers)),
            ]
            wrist_viz = draw_obs_from_array(wrist_raw, raw_obs, extra_labels=extra_labels)
            wrist_bgr = cv2.cvtColor(wrist_viz, cv2.COLOR_RGB2BGR)
            cv2.putText(wrist_bgr, "Wrist cam (obs)", (8, wrist_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Composite frame (2 top, 3 bottom) ────────────────────────────
            PANEL_W = 480
            PANEL_H = 360

            def fit_panel(img):
                h, w = img.shape[:2]
                scale_f = min(PANEL_W / w, PANEL_H / h)
                nw, nh = int(w * scale_f), int(h * scale_f)
                resized = cv2.resize(img, (nw, nh))
                out = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                y0 = (PANEL_H - nh) // 2
                x0 = (PANEL_W - nw) // 2
                out[y0:y0+nh, x0:x0+nw] = resized
                return out

            sep_v = np.full((PANEL_H, 4, 3), 60, dtype=np.uint8)

            sep_v3  = np.full((PANEL_H, 4, 3), 60, dtype=np.uint8)
            top_row = np.hstack([fit_panel(disp), sep_v, fit_panel(sim_bgr)])
            bot_row = np.hstack([fit_panel(rect), sep_v3, fit_panel(top_bgr), sep_v3, fit_panel(wrist_bgr)])

            # Pad narrower row to match wider row
            pad_w = bot_row.shape[1] - top_row.shape[1]
            if pad_w > 0:
                top_row = np.hstack([top_row, np.zeros((PANEL_H, pad_w, 3), dtype=np.uint8)])
            elif pad_w < 0:
                bot_row = np.hstack([bot_row, np.zeros((PANEL_H, -pad_w, 3), dtype=np.uint8)])

            sep_h_wide = np.full((4, top_row.shape[1], 3), 60, dtype=np.uint8)
            panels = np.vstack([top_row, sep_h_wide, bot_row])

            bar_h = 32
            bar   = np.zeros((bar_h, panels.shape[1], 3), dtype=np.uint8)
            if block_sim_pos:
                debug = (f"step={step}  int=({cx_int:.1f},{cy_int:.1f})mm  |  "
                         f"sim=({block_sim_pos[0]:.3f},{block_sim_pos[1]:.3f},{block_sim_pos[2]:.3f})m  |  "
                         f"inliers={n_inliers}  ROBOT_INT_X={ROBOT_INT_X:.0f}  "
                         f"ROBOT_INT_Y={ROBOT_INT_Y:.0f}  BASE_TIP_X={ROBOT_BASE_TIP_X:.3f}")
            else:
                debug = f"step={step}  ID {args.block_tag} not detected  |  inliers={n_inliers}"
            cv2.putText(bar, debug, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
            combined = np.vstack([bar, panels])

            if writer is None:
                out_h, out_w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
                print(f"Writer ready: {out_w}×{out_h}")

            writer.write(combined)
            frame_count += 1

            if done[0]:
                print(f"Episode done at step {step}")
                break

            elapsed = time.time() - t_start
            sleep_t = max(0.0, 1.0 / args.fps - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\nStopped at step {frame_count}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved {args.output}")
        if robot is not None:
            robot.disable_torque()
            robot.disconnect()
        wrist_renderer.close()
        policy_env.close()
        viz_env.close()


if __name__ == "__main__":
    main()
