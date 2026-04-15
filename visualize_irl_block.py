"""visualize_irl_block.py — Show real-world block in MuJoCo at its detected position.

Detects a bordered ArUco tag on the block via the overhead camera, applies
perspective correction to get the board-mm position, converts to simulation
coordinates, and moves the cube body in the MuJoCo viewer in real time.

Physical setup assumption:
    - The calibration board (180mm square) is flat on the table
    - The robot arm base outer-bottom tip sits at the bottom-center of the board
    - The block carries bordered ArUco tag ID 101 on top (one jenga block, h=15mm)

Coordinate transform  (interior mm → simulation metres):
    sim_x =  (ROBOT_INT_Y - interior_y) / 1000    # forward distance from robot
    sim_y = -(interior_x  - ROBOT_INT_X) / 1000   # lateral (right of robot = –y)

If the block appears in the wrong place in sim, tweak ROBOT_INT_X / ROBOT_INT_Y
or flip the sign of sim_y.

Usage:
    uv run python visualize_irl_block.py
    uv run python visualize_irl_block.py --camera 1 --block-height 0 --block-tag 102
    uv run python visualize_irl_block.py --camera 1 --block-height 30
"""

import argparse
import atexit
import tempfile
from pathlib import Path

import cv2
import cv2.aruco as aruco
import mujoco
import numpy as np

from src.envs.lift_cube import LiftCubeCartesianEnv

# ── ArUco detection parameters (must match test_aruco_homography_3d.py) ──────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.errorCorrectionRate       = 0.7

BORDERED_IDS = set(range(100, 200))
MARGIN_FRAC  = 48 / 200   # outer border relative to inner ArUco side

# ── Manual position offset (tune these to correct block position) ─────────────
DELTA_X =  0.0   # mm to add to observed X
DELTA_Y =  -10   # mm to add to observed Y

# ── Board geometry ────────────────────────────────────────────────────────────
SQUARE_MM = 180.0
TAG_MM    = 16.0
GAP_MM    = 2.0
N_OUTER   = 10
N_INNER   = 8
INSET_MM  = 2 * TAG_MM + GAP_MM   # = 34mm — board-corner to interior-corner

OUT_W = 600
OUT_H = 600

# ── Robot base position in interior mm ───────────────────────────────────────
# The robot arm outer-bottom-tip sits at board coords (90, 180) mm
# (bottom-center of the 180mm board).  Converting to interior mm:
#   interior = board − INSET_MM  →  (90−34, 180−34) = (56, 146)
#
# ADJUST THESE if the simulated block appears in the wrong place:
ROBOT_INT_X = 56.0    # interior mm — lateral position of robot base centre
ROBOT_INT_Y = 146.0   # interior mm — forward edge of robot base

# ── Tuning constants — adjust these if the sim block position is wrong ────────
#
# ROBOT_BASE_TIP_X : how far the FRONT FACE of the robot base is from the
#                    world origin (m).  Physical base is 11cm → front at 5.5cm.
#                    Increase to push the board further forward.
ROBOT_BASE_TIP_X = 0.055  # metres  ← tune here
#
# ROBOT_INT_X : interior-mm X coordinate of the robot base centre-line on the
#               board.  Board is 180mm wide, interior inset 34mm each side,
#               so centre = 90-34 = 56mm.  Adjust left/right if block drifts
#               sideways in sim vs panel 3.
#   (set near top of file, currently ROBOT_INT_X = 56.0)
#
# ROBOT_INT_Y : interior-mm Y coordinate of the board's back edge (where the
#               robot base tip touches the board).  Board is 180mm, inset 34mm,
#               so outer back edge = 180-34 = 146mm interior.  Adjust forward/
#               back if block drifts in the sim X direction vs panel 3.
#   (set near top of file, currently ROBOT_INT_Y = 146.0)

# ── Simulation cube size ──────────────────────────────────────────────────────
CUBE_HALF = 0.015   # metres — must match lift_cube.xml geom size


# ─────────────────────────────────────────────────────────────────────────────
# Board helper functions (mirrored from test_aruco_homography_3d.py)
# ─────────────────────────────────────────────────────────────────────────────

def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    """Return {tag_id: (x_mm, y_mm)} for all border tags.
    Origin = top-left corner of the board square.  X → right, Y → down."""
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
    """Build image→board-mm homography from detected border tags."""
    src_pts, dst_pts = [], []
    for tag_id, img_corners in corners_dict.items():
        if tag_id not in tag_centres_mm:
            continue
        cx_img = float(img_corners[:, 0].mean())
        cy_img = float(img_corners[:, 1].mean())
        cx_mm, cy_mm = tag_centres_mm[tag_id]
        src_pts.append([cx_img, cy_img])
        dst_pts.append([cx_mm,  cy_mm])
    if len(src_pts) < 4:
        return None, len(src_pts)
    src    = np.array(src_pts, dtype=np.float32)
    dst    = np.array(dst_pts, dtype=np.float32)
    scale  = OUT_W / square_mm
    dst_px = dst * scale
    H, mask = cv2.findHomography(src, dst_px, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def bordered_corners_image(img_corners):
    """Extend inner ArUco corners to full bordered-tag footprint in image px."""
    tl, tr, br, bl = img_corners.astype(np.float64)
    right = tr - tl
    down  = bl - tl
    m     = MARGIN_FRAC
    return np.array([
        tl - m * right - m * down,
        tr + m * right - m * down,
        br + m * right + m * down,
        bl - m * right + m * down,
    ], dtype=np.float64)


def get_interior_pos(img_corners, H, scale, inset_mm):
    """Return (x_int, y_int) interior mm of the tag TL corner."""
    tl_img = img_corners[0].astype(np.float32)
    board_pt = cv2.perspectiveTransform(tl_img.reshape(1, 1, 2), H).reshape(2)
    return float(board_pt[0]) / scale - inset_mm, float(board_pt[1]) / scale - inset_mm


def perspective_correct(obs_x, obs_y, h_mm=None):
    return obs_x + DELTA_X, obs_y + DELTA_Y


CUBE_HALF_MM = CUBE_HALF * 1000  # 15mm — half the cube side in interior coords


def interior_to_sim(int_x_mm, int_y_mm, height_mm):
    """Convert tag-centre interior mm position to simulation cube-centre metres."""
    sim_x = ROBOT_BASE_TIP_X + (ROBOT_INT_Y - int_y_mm) / 1000.0
    sim_y = -(int_x_mm - ROBOT_INT_X) / 1000.0
    sim_z = CUBE_HALF
    return sim_x, sim_y, sim_z


# ─────────────────────────────────────────────────────────────────────────────
# Rectified top-down view (third panel)
# ─────────────────────────────────────────────────────────────────────────────

def draw_rectified(frame, H, square_mm, tag_mm, gap_mm, corners_dict,
                   tag_centres_mm, block_tag, block_height, scale, inset_mm):
    """Return a bird's-eye warped view of the board with the block marked."""
    if H is None:
        blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No homography", (20, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return blank

    warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))

    interior_size_mm = square_mm - 2 * inset_mm
    inset_px  = int(inset_mm * scale)
    isize_px  = int(interior_size_mm * scale)

    # Outer border
    cv2.rectangle(warped, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)

    # Interior border
    cv2.rectangle(warped,
                  (inset_px, inset_px),
                  (inset_px + isize_px, inset_px + isize_px),
                  (0, 0, 0), 2)

    # Grid lines every 10mm
    for d_mm in range(10, int(interior_size_mm), 10):
        d_px  = inset_px + int(d_mm * scale)
        color = (80, 80, 80) if d_mm % 20 == 0 else (160, 160, 160)
        cv2.line(warped, (d_px, inset_px), (d_px, inset_px + isize_px), color, 1)
        cv2.line(warped, (inset_px, d_px), (inset_px + isize_px, d_px), color, 1)

    # Border tag dots
    for tid, (cx_mm, cy_mm) in tag_centres_mm.items():
        cx_px = int(cx_mm * scale)
        cy_px = int(cy_mm * scale)
        cv2.circle(warped, (cx_px, cy_px), 3, (0, 160, 0), -1)

    # Block tag
    if block_tag in corners_dict:
        img_corners = corners_dict[block_tag]
        b_img = bordered_corners_image(img_corners)
        b_board = cv2.perspectiveTransform(
            b_img.astype(np.float32).reshape(1, -1, 2), H).reshape(-1, 2)

        # Raw TL position
        tl_px = b_board[0].astype(int)
        x_int = float(b_board[0][0]) / scale - inset_mm
        y_int = float(b_board[0][1]) / scale - inset_mm

        # Draw bordered tag outline (raw)
        cv2.polylines(warped, [b_board.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 200, 255), 2)
        cv2.drawMarker(warped, tuple(tl_px), (0, 200, 255),
                       cv2.MARKER_CROSS, 14, 2)

        # Perspective-corrected position (green dot)
        cx_int, cy_int = perspective_correct(x_int, y_int, block_height)
        cx_px = inset_px + int(cx_int * scale)
        cy_px = inset_px + int(cy_int * scale)
        cv2.circle(warped, (cx_px, cy_px), 7, (0, 255, 0), -1)
        cv2.putText(warped,
                    f"({cx_int:.1f},{cy_int:.1f})mm",
                    (cx_px + 8, cy_px - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return warped


# ─────────────────────────────────────────────────────────────────────────────
# Board scene builder
# ─────────────────────────────────────────────────────────────────────────────

def board_to_sim(board_x_mm, board_y_mm):
    """Convert board mm coords (from board top-left) to simulation metres.

    Board back edge (board_y = SQUARE_MM) aligns to ROBOT_BASE_TIP_X.
    """
    sim_x = ROBOT_BASE_TIP_X + (SQUARE_MM - board_y_mm) / 1000.0
    sim_y = -(board_x_mm - SQUARE_MM / 2) / 1000.0
    return sim_x, sim_y


def build_scene_xml(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    """Write a temp scene XML that includes lift_cube.xml + calibration board geoms.

    Returns the path to the temp file (caller should delete when done).
    """
    tag_half  = tag_mm / 2 / 1000        # metres half-size per tag
    board_h   = square_mm / 2 / 1000     # board half-size (square)
    inset_mm  = 2 * tag_mm + gap_mm
    int_half  = (square_mm / 2 - inset_mm) / 1000   # interior half-size

    board_cx, board_cy = board_to_sim(square_mm / 2, square_mm / 2)

    geoms = []

    # Board surface (cardboard colour)
    geoms.append(
        f'<geom name="board_surface" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} -0.0004" '
        f'size="{board_h:.4f} {board_h:.4f} 0.0004" '
        f'rgba="0.85 0.80 0.65 1" contype="0" conaffinity="0" group="1"/>'
    )

    # Interior region (lighter inset)
    geoms.append(
        f'<geom name="board_interior" type="box" '
        f'pos="{board_cx:.4f} {board_cy:.4f} 0.00005" '
        f'size="{int_half:.4f} {int_half:.4f} 0.00005" '
        f'rgba="0.92 0.88 0.72 1" contype="0" conaffinity="0" group="1"/>'
    )

    # Border tags — dark squares at each tag centre
    tag_centres = board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner)
    for i, (bx, by) in tag_centres.items():
        sx, sy = board_to_sim(bx, by)
        geoms.append(
            f'<geom name="btag_{i}" type="box" '
            f'pos="{sx:.4f} {sy:.4f} 0.0001" '
            f'size="{tag_half:.4f} {tag_half:.4f} 0.0001" '
            f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
        )

    # Robot base tip line — thin red stripe across the board at ROBOT_BASE_TIP_X
    geoms.append(
        f'<geom name="base_tip_line" type="box" '
        f'pos="{ROBOT_BASE_TIP_X:.4f} 0.0000 0.0003" '
        f'size="0.0008 {board_h:.4f} 0.0003" '
        f'rgba="1 0.15 0.15 1" contype="0" conaffinity="0" group="1"/>'
    )

    geoms_xml = "\n        ".join(geoms)

    # Top-down camera centred over the board
    board_cx, board_cy = board_to_sim(square_mm / 2, square_mm / 2)
    topdown_cam = (
        f'<camera name="topdown" pos="{board_cx:.4f} {board_cy:.4f} 0.6" '
        f'euler="0 0 3.14159" fovy="30"/>'
    )

    # Wrap in a scene XML that includes the base model
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
    parser.add_argument("--camera",       type=int,   default=0,
                        help="Camera device index")
    parser.add_argument("--block-tag",    type=int,   default=101,
                        help="Bordered ArUco tag ID on the block (default 101)")
    parser.add_argument("--block-height", type=float, default=15.0,
                        help="Block tag height above board in mm (default 15)")
    parser.add_argument("--square-mm",    type=float, default=SQUARE_MM)
    parser.add_argument("--tag-mm",       type=float, default=TAG_MM)
    parser.add_argument("--gap-mm",       type=float, default=GAP_MM)
    parser.add_argument("--n-outer",      type=int,   default=N_OUTER)
    parser.add_argument("--n-inner",      type=int,   default=N_INNER)
    parser.add_argument("--output",       type=str,   default="irl_block.mp4")
    parser.add_argument("--fps",          type=float, default=30.0)
    args = parser.parse_args()

    inset_mm = 2 * args.tag_mm + args.gap_mm
    scale    = OUT_W / args.square_mm

    tag_centres_mm = board_tag_centres(
        args.square_mm, args.tag_mm, args.gap_mm, args.n_outer, args.n_inner)

    print(f"Board: {args.square_mm:.0f}mm  |  {len(tag_centres_mm)} border tags")
    print(f"Interior inset: {inset_mm:.1f}mm")
    print(f"Tracking block tag ID={args.block_tag}  h={args.block_height:.0f}mm")
    print(f"Robot base at interior ({ROBOT_INT_X:.0f}, {ROBOT_INT_Y:.0f}) mm")
    print("Press Q in the camera window to quit")

    # ── Build scene with calibration board, load env ─────────────────────────
    scene_xml = build_scene_xml(args.square_mm, args.tag_mm, args.gap_mm,
                                args.n_outer, args.n_inner)
    env = LiftCubeCartesianEnv(render_mode="rgb_array", scene_path=scene_xml)
    env.reset()

    cube_jnt_id   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_adr = env.model.jnt_qposadr[cube_jnt_id]

    # Park the cube below the floor until first detection
    env.data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = [0.0, 0.0, -1.0]
    env.data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    H        = None

    detected_last = False
    cx_int = cy_int = 0.0
    n_inliers = 0
    block_sim_pos = None
    frame_count = 0
    writer = None

    output = args.output
    print(f"Recording to {output}  (Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det_corners, det_ids, _ = detector.detectMarkers(gray)

            corners_dict = {}
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    corners_dict[int(tid)] = det_corners[i].reshape(4, 2)

            H, n_inliers = compute_homography(corners_dict, tag_centres_mm, args.square_mm)

            # Keep last known position if detection is lost
            if H is not None and args.block_tag in corners_dict:
                img_corners = corners_dict[args.block_tag]
                x_int, y_int = get_interior_pos(img_corners, H, scale, inset_mm)
                cx_int, cy_int = perspective_correct(x_int, y_int, args.block_height)
                sim_x, sim_y, sim_z = interior_to_sim(cx_int, cy_int, args.block_height)
                block_sim_pos = (sim_x, sim_y, sim_z)
                if not detected_last:
                    print(f"[{frame_count}] Block detected  "
                          f"interior=({cx_int:.1f},{cy_int:.1f})mm  "
                          f"sim=({sim_x:.3f},{sim_y:.3f},{sim_z:.3f})m")
                detected_last = True
            else:
                if detected_last:
                    print(f"[{frame_count}] Block lost")
                detected_last = False

            # Update cube in sim (block_sim_pos retains last known if detection lost)
            if block_sim_pos is not None:
                env.data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = block_sim_pos
            env.data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]
            env.data.qvel[cube_qpos_adr:cube_qpos_adr + 6] = 0
            mujoco.mj_forward(env.model, env.data)

            # Render sim via env.render() — same as eval_cartesian.py
            sim_rgb = env.render(camera="wide")   # (H, W, 3) RGB
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

            # ── Panel 1: annotated camera frame ──────────────────────────────
            disp = frame.copy()
            if det_ids is not None:
                for i, tid in enumerate(det_ids.flatten()):
                    pts = det_corners[i].reshape(4, 2).astype(np.int32)
                    is_border = int(tid) in tag_centres_mm
                    is_block  = int(tid) == args.block_tag
                    color = (0, 255, 0) if is_border else (0, 200, 255) if is_block else (80, 80, 255)
                    cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, color, 2)
                    cx_px, cy_px = pts.mean(axis=0).astype(int)
                    cv2.putText(disp, str(tid), (cx_px - 8, cy_px + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            status = f"inliers={n_inliers} | "
            status += (f"int=({cx_int:.1f},{cy_int:.1f})mm"
                       if block_sim_pos else f"ID {args.block_tag} not detected")
            cv2.putText(disp, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(disp, "Camera", (8, disp.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 2: MuJoCo simulation (side view) ───────────────────────
            cv2.putText(sim_bgr, "Simulation (side)", (8, sim_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 3: rectified top-down board view ────────────────────────
            rect = draw_rectified(frame, H, args.square_mm, args.tag_mm,
                                  args.gap_mm, corners_dict, tag_centres_mm,
                                  args.block_tag, args.block_height, scale, inset_mm)
            cv2.putText(rect, "Top-down (homography)", (8, rect.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Panel 4: top-down sim camera, arm hidden ──────────────────────
            if env._renderer is None:
                env._renderer = mujoco.Renderer(env.model, height=480, width=640)
            opt = mujoco.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[0] = 1   # floor + cube (default group)
            opt.geomgroup[1] = 1   # board geoms (group="1" in XML)
            env._renderer.update_scene(env.data, camera="topdown", scene_option=opt)
            top_rgb = env._renderer.render()
            top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(top_bgr, "Simulation (top-down)", (8, top_bgr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # ── Debug bar above the panels ────────────────────────────────────
            target_h = sim_bgr.shape[0]
            sep = np.full((target_h, 4, 3), 60, dtype=np.uint8)

            def fit_h(img):
                h, w = img.shape[:2]
                return cv2.resize(img, (int(w * target_h / h), target_h))

            panels = np.hstack([fit_h(disp), sep, sim_bgr, sep, fit_h(rect), sep, fit_h(top_bgr)])
            bar_h  = 32
            bar    = np.zeros((bar_h, panels.shape[1], 3), dtype=np.uint8)
            if block_sim_pos:
                debug = (f"interior=({cx_int:.1f}, {cy_int:.1f}) mm  |  "
                         f"sim=({block_sim_pos[0]:.3f}, {block_sim_pos[1]:.3f}, {block_sim_pos[2]:.3f}) m  |  "
                         f"ROBOT_INT_X={ROBOT_INT_X:.0f}  ROBOT_INT_Y={ROBOT_INT_Y:.0f}  "
                         f"BASE_TIP_X={ROBOT_BASE_TIP_X:.3f}")
            else:
                debug = f"ID {args.block_tag} not detected  |  inliers={n_inliers}"
            cv2.putText(bar, debug, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
            combined = np.vstack([bar, panels])

            if writer is None:
                out_h, out_w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(output, fourcc, args.fps, (out_w, out_h))
                print(f"Writer ready: {out_w}×{out_h} @ {args.fps:.0f}fps")

            writer.write(combined)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  {frame_count} frames...")

    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames")
    finally:
        cap.release()
        env.close()
        if writer is not None:
            writer.release()
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
