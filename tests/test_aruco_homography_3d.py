"""test_aruco_homography_3d.py — Live homography viewer + perspective calibration.

Opens the camera, detects border tags on the printed board, computes a
homography from the tag layout to a canonical flat-board view, and shows:
  Left  — raw camera feed with detected tags highlighted
  Right — rectified bird's-eye view of the board interior

Calibration mode (press C):
    Place ONE jenga block (tag ID 101) at each of the 4 interior corners.
    The block corner should touch the interior corner each time.
    The script uses the ArUco tag CENTRE as the position measurement.
    After all 4 corners are recorded it fits (Cx, Cy, Cz) and prints
    the values to paste into visualize_irl_block.py and run_real_t1.py.

Usage:
    uv run python test_aruco_homography_3d.py
    uv run python test_aruco_homography_3d.py --camera 1 --block-tag 101 \\
        --block-height 15 --block-w 25 --block-d 75
"""
import argparse
import time

import cv2
import cv2.aruco as aruco
import numpy as np

# ── Board geometry (must match make_aruco_board.py defaults) ─────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
# Slightly more aggressive detection: larger adaptive threshold windows and
# a touch more error correction tolerance.
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 53   # default 23 — tries more window sizes
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4    # default 10 — finer steps
ARUCO_PARAMS.errorCorrectionRate       = 0.7  # default 0.6 — slightly more lenient

# Output panel size for the rectified view
OUT_W = 600
OUT_H = 600

# ── Bordered tag IDs (only these are treated as objects on the board) ─────────
# Must match the IDs printed by make_bordered_tags.py.
# Board border tags are always excluded separately (they're in tag_centres_mm).
BORDERED_IDS = set(range(100, 200))

# ── Bordered tag geometry ──────────────────────────────────────────────────────
# From bordered_aruco.py default params (at tag_size_px=200):
#   margin_per_side = border(10) + double_gap(8) + double_border(6) + quiet(24) = 48
#   MARGIN_FRAC = 48 / 200 = 0.24
# Used to reconstruct the full bordered tag footprint from the inner ArUco corners.
MARGIN_FRAC = 48 / 200

# ── Manual position offset (tune these to correct block position) ─────────────
DELTA_X =  -10   # mm to add to observed X
DELTA_Y =  1   # mm to add to observed Y


def correct_pos(obs_x, obs_y):
    """Apply manual offset to observed interior position."""
    return obs_x + DELTA_X, obs_y + DELTA_Y


# ── Calibration constants ─────────────────────────────────────────────────────
# Corner labels and order for the 4-corner calibration
CALIB_CORNER_NAMES = ["TL", "TR", "BL", "BR"]


def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    """Return {tag_id: (x_mm, y_mm)} for all border tags.

    Origin = top-left corner of the square.  X → right, Y → down.
    """
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
    """Build src (image pixels) → dst (mm on board) correspondences.

    Returns H (3×3) or None, plus inlier count.
    """
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

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    scale  = OUT_W / square_mm
    dst_px = dst * scale

    H, mask = cv2.findHomography(src, dst_px, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def bordered_corners_image(img_corners):
    """Compute the 4 corners of the full bordered tag in image pixel space.

    The ArUco detector returns corners of the inner bit-grid only.
    The bordered tag extends MARGIN_FRAC * side_px further on every side.

    img_corners : (4, 2) float — ArUco corners in detection order [TL, TR, BR, BL]
    Returns     : (4, 2) float — bordered tag corners [TL, TR, BR, BL]
    """
    tl, tr, br, bl = img_corners.astype(np.float64)
    right = tr - tl   # along top edge (length = ArUco side in image px)
    down  = bl - tl   # along left edge
    m     = MARGIN_FRAC
    return np.array([
        tl - m * right - m * down,   # bordered TL
        tr + m * right - m * down,   # bordered TR
        br + m * right + m * down,   # bordered BR
        bl - m * right + m * down,   # bordered BL
    ], dtype=np.float64)


def tag_interior_pos(img_corners, H, scale, interior_inset_mm):
    """Return the TL corner (x_int, y_int) of the ArUco tag in interior mm."""
    tl_img = img_corners[0].astype(np.float32)
    board_pt = cv2.perspectiveTransform(tl_img.reshape(1, 1, 2), H).reshape(2)
    return (float(board_pt[0]) / scale - interior_inset_mm,
            float(board_pt[1]) / scale - interior_inset_mm)


def draw_raw_panel(frame, corners_dict, tag_centres_mm, H, square_mm):
    out = frame.copy()
    for tag_id, corners in corners_dict.items():
        pts   = corners.astype(np.int32).reshape((-1, 1, 2))
        known = tag_id in tag_centres_mm
        color = (0, 220, 0) if known else (0, 100, 255)
        cv2.polylines(out, [pts], True, color, 2)
        cx = int(corners[:, 0].mean())
        cy = int(corners[:, 1].mean())
        cv2.putText(out, str(tag_id), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    if H is not None:
        H_inv = np.linalg.inv(H)
        scale = OUT_W / square_mm
        brd   = np.array([[0, 0], [square_mm, 0],
                           [square_mm, square_mm], [0, square_mm]], dtype=np.float32)
        brd_out_px = brd * scale
        brd_img    = cv2.perspectiveTransform(
            brd_out_px.reshape(1, -1, 2), H_inv).reshape(-1, 2)
        cv2.polylines(out, [brd_img.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 200, 255), 2)
    return out


def draw_rectified_panel(frame, H, square_mm, tag_mm, gap_mm,
                          corners_dict, tag_centres_mm, inner_tags_start,
                          calib_state):
    """Warp frame to bird's-eye view; overlay grid, known tag dots, inner objects,
    and calibration guide overlays."""
    if H is None:
        blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(blank, "Need >= 4 border tags", (20, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return blank

    warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))
    scale  = OUT_W / square_mm

    interior_inset_mm = 2 * tag_mm + gap_mm
    interior_size_mm  = square_mm - 2 * interior_inset_mm
    inset_px = int(interior_inset_mm * scale)
    isize_px = int(interior_size_mm * scale)

    # Outer square border
    cv2.rectangle(warped, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)

    # Interior border
    cv2.rectangle(warped,
                  (inset_px, inset_px),
                  (inset_px + isize_px, inset_px + isize_px),
                  (0, 0, 0), 2)

    # Grid lines
    for d_mm in range(5, int(interior_size_mm), 5):
        d_px     = inset_px + int(d_mm * scale)
        is_major = d_mm % 10 == 0
        color    = (100, 100, 100) if is_major else (180, 180, 180)
        cv2.line(warped, (d_px, inset_px), (d_px, inset_px + isize_px), color, 1)
        cv2.line(warped, (inset_px, d_px), (inset_px + isize_px, d_px), color, 1)
        if is_major:
            lbl = str(d_mm)
            cv2.putText(warped, lbl, (d_px + 2, inset_px + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(warped, lbl, (inset_px + 2, d_px - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

    # Border tag positions
    for tag_id, (cx_mm, cy_mm) in tag_centres_mm.items():
        cx_px = int(cx_mm * scale)
        cy_px = int(cy_mm * scale)
        color = (0, 180, 0) if tag_id < inner_tags_start else (0, 120, 200)
        cv2.circle(warped, (cx_px, cy_px), 4, color, -1)

    # Inner bordered tags (non-border, bordered-ID only)
    for tag_id, corners in corners_dict.items():
        if tag_id in tag_centres_mm:
            continue
        if tag_id not in BORDERED_IDS:
            continue

        # Warp the full bordered tag outline into board space
        b_img   = bordered_corners_image(corners)
        b_board = cv2.perspectiveTransform(
            b_img.astype(np.float32).reshape(1, -1, 2), H).reshape(-1, 2)
        tl_ipt = (int(b_board[0, 0]), int(b_board[0, 1]))

        # Tag TL corner in interior mm (raw observed)
        tl_img = corners[0].astype(np.float32)
        tl_board = cv2.perspectiveTransform(
            tl_img.reshape(1, 1, 2), H).reshape(2)
        x_int = float(tl_board[0]) / scale - interior_inset_mm
        y_int = float(tl_board[1]) / scale - interior_inset_mm
        # raw_px for marker: TL corner in warped space
        centre_board = tl_board

        is_calib = (tag_id == calib_state.get('block_tag', 101))
        color    = (0, 200, 255) if is_calib else (0, 0, 255)

        # Draw the full bordered tag outline
        cv2.polylines(warped, [b_board.astype(np.int32).reshape(-1, 1, 2)],
                      True, color, 2)

        # Raw centre: centre_board is already in warped pixel space
        raw_px = (int(centre_board[0]), int(centre_board[1]))
        cv2.drawMarker(warped, raw_px, color, cv2.MARKER_CROSS, 14, 2)

        # Corrected position — convert interior mm back to warped pixel
        cx_int, cy_int = correct_pos(x_int, y_int)
        cx_px = int((cx_int + interior_inset_mm) * scale)
        cy_px = int((cy_int + interior_inset_mm) * scale)
        cv2.circle(warped, (cx_px, cy_px), 6, (0, 255, 0), -1)
        cv2.putText(warped,
                    f"ID{tag_id}  raw=({x_int:.1f},{y_int:.1f})  "
                    f"corr=({cx_int:.1f},{cy_int:.1f})mm",
                    (raw_px[0] + 4, raw_px[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # ── Calibration overlays ──────────────────────────────────────────────────
    if calib_state['active']:
        step = calib_state['step']
        corner_positions = calib_state.get('corner_positions', [])
        bw_px = int(calib_state.get('block_w', 25) * scale)
        bd_px = int(calib_state.get('block_d', 75) * scale)
        # Corner touch points in interior coords (0,0), (iW,0), (0,iH), (iW,iH)
        isize_px = int((square_mm - 2 * (2 * tag_mm + gap_mm)) * scale)
        corner_touch_px = [
            (inset_px,           inset_px),            # TL
            (inset_px + isize_px, inset_px),           # TR
            (inset_px,           inset_px + isize_px), # BL
            (inset_px + isize_px, inset_px + isize_px),# BR
        ]
        # Block rect offsets from touch corner [TL, TR, BL, BR]
        block_rects = [
            (0,      0,      bw_px, bd_px),  # TL: extends right+down
            (-bw_px, 0,      0,     bd_px),  # TR: extends left+down
            (0,      -bd_px, bw_px, 0),      # BL: extends right+up
            (-bw_px, -bd_px, 0,     0),      # BR: extends left+up
        ]

        if step < len(corner_positions):
            true_x, true_y = corner_positions[step]
            tx_px = inset_px + int(true_x * scale)
            ty_px = inset_px + int(true_y * scale)
            # Draw block outline at target position
            cx_touch, cy_touch = corner_touch_px[step]
            dx0, dy0, dx1, dy1 = block_rects[step]
            cv2.rectangle(warped,
                          (cx_touch + dx0, cy_touch + dy0),
                          (cx_touch + dx1, cy_touch + dy1),
                          (0, 255, 255), 2)
            # Cross at tag centre
            cv2.drawMarker(warped, (tx_px, ty_px), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)
            label = CALIB_CORNER_NAMES[step]
            cv2.putText(warped, label, (cx_touch + 4, cy_touch + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Already-recorded corners
        for idx, (px, py) in enumerate(calib_state.get('done_px', [])):
            cv2.circle(warped, (px, py), 6, (0, 220, 0), -1)
            cv2.putText(warped, CALIB_CORNER_NAMES[idx], (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)

    return warped


def fit_linear_2d(calib_data):
    """Fit per-axis linear correction from 4 corner measurements.

    calib_data : list of (true_x, true_y, obs_x, obs_y, height_mm)
    Model: true_x = Sx * obs_x + Bx
           true_y = Sy * obs_y + By
    """
    arr = np.array(calib_data, dtype=np.float64)
    true_x = arr[:, 0]; true_y = arr[:, 1]
    obs_x  = arr[:, 2]; obs_y  = arr[:, 3]

    # Least-squares fit: [obs  1] @ [S, B]^T = true
    A_x = np.column_stack([obs_x, np.ones(len(obs_x))])
    A_y = np.column_stack([obs_y, np.ones(len(obs_y))])
    Sx, Bx = np.linalg.lstsq(A_x, true_x, rcond=None)[0]
    Sy, By = np.linalg.lstsq(A_y, true_y, rcond=None)[0]

    pred_x = Sx * obs_x + Bx
    pred_y = Sy * obs_y + By
    rms = np.sqrt(np.mean((pred_x - true_x)**2 + (pred_y - true_y)**2))

    print("\n" + "="*60)
    print("2-D LINEAR CALIBRATION RESULTS")
    print("="*60)
    print(f"{'Corner':>6}  {'True X':>7}  {'True Y':>7}  {'Obs X':>7}  {'Obs Y':>7}  "
          f"{'Err X':>7}  {'Err Y':>7}")
    for i, (tx, ty, ox, oy, _) in enumerate(arr):
        px = Sx * ox + Bx
        py = Sy * oy + By
        print(f"{CALIB_CORNER_NAMES[i]:>6}  {tx:7.2f}  {ty:7.2f}  {ox:7.2f}  {oy:7.2f}  "
              f"{px-tx:+7.2f}  {py-ty:+7.2f}")
    print(f"\nRMS residual: {rms:.2f} mm")
    print(f"\n# Paste these into visualize_irl_block.py and run_real_t1.py:")
    print(f"CORR_Sx = {Sx:8.5f}")
    print(f"CORR_Bx = {Bx:8.3f}")
    print(f"CORR_Sy = {Sy:8.5f}")
    print(f"CORR_By = {By:8.3f}")
    print("="*60)
    return Sx, Bx, Sy, By


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",       type=int,   default=0)
    parser.add_argument("--square-mm",    type=float, default=180.0)
    parser.add_argument("--tag-mm",       type=float, default=16.0)
    parser.add_argument("--gap-mm",       type=float, default=2.0)
    parser.add_argument("--n-outer",      type=int,   default=10)
    parser.add_argument("--n-inner",      type=int,   default=8)
    parser.add_argument("--block-tag",    type=int,   default=101,
                        help="ArUco ID on top of the jenga block (default 101)")
    parser.add_argument("--block-height", type=float, default=15.0,
                        help="Block height in mm (default 15)")
    parser.add_argument("--block-w",      type=float, default=25.0,
                        help="Block width in mm (default 25)")
    parser.add_argument("--block-d",      type=float, default=75.0,
                        help="Block depth in mm (default 75)")
    args = parser.parse_args()

    tag_centres_mm    = board_tag_centres(
        args.square_mm, args.tag_mm, args.gap_mm, args.n_outer, args.n_inner)
    n_outer_total     = args.n_outer * 2 + (args.n_outer - 2) * 2
    inner_tags_start  = n_outer_total
    interior_inset_mm = 2 * args.tag_mm + args.gap_mm
    scale             = OUT_W / args.square_mm
    interior_size_mm  = args.square_mm - 2 * interior_inset_mm

    # True tag-centre positions when block corner touches each interior corner.
    # Block is oriented so its shorter dimension (block_w) faces X and longer (block_d) faces Y.
    hw = args.block_w / 2
    hd = args.block_d / 2
    iW = interior_size_mm
    iH = interior_size_mm
    # [TL, TR, BL, BR] — tag centre = block centre
    corner_positions = [
        (hw,      hd),        # TL
        (iW - hw, hd),        # TR
        (hw,      iH - hd),   # BL
        (iW - hw, iH - hd),   # BR
    ]

    print(f"Board: {args.square_mm:.0f}mm  |  {len(tag_centres_mm)} border tags")
    print(f"Interior inset: {interior_inset_mm:.1f}mm  |  Interior size: {interior_size_mm:.1f}mm")
    print(f"Block: {args.block_w}×{args.block_d}mm  h={args.block_height}mm  tag ID={args.block_tag}")
    print(f"Corner positions (tag centre): {[f'({x:.1f},{y:.1f})' for x,y in corner_positions]}")
    print("Keys:  Q = quit   S = snapshot   C = start/reset calibration   SPACE = record corner")

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    cap      = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    H = None

    # Calibration state — simple 4-step counter
    calib = {
        'active':           False,
        'step':             0,
        'data':             [],   # (true_x, true_y, obs_x, obs_y, height_mm)
        'done_px':          [],
        'corner_positions': corner_positions,
        'done':             False,
        'block_tag':        args.block_tag,
        'block_w':          args.block_w,
        'block_d':          args.block_d,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_corners, det_ids, _ = detector.detectMarkers(gray)

        corners_dict = {}
        if det_ids is not None:
            for i, tag_id in enumerate(det_ids.flatten()):
                corners_dict[int(tag_id)] = det_corners[i].reshape(4, 2)

        H, n_inliers = compute_homography(
            corners_dict, tag_centres_mm, args.square_mm)

        raw_panel  = draw_raw_panel(frame, corners_dict, tag_centres_mm, H, args.square_mm)
        rect_panel = draw_rectified_panel(
            frame, H, args.square_mm, args.tag_mm, args.gap_mm,
            corners_dict, tag_centres_mm, inner_tags_start, calib)

        # Resize panels to same height
        ph = max(raw_panel.shape[0], rect_panel.shape[0])
        raw_panel  = cv2.resize(raw_panel,  (int(raw_panel.shape[1]  * ph / raw_panel.shape[0]),  ph))
        rect_panel = cv2.resize(rect_panel, (int(rect_panel.shape[1] * ph / rect_panel.shape[0]), ph))

        combined = np.hstack([raw_panel, np.full((ph, 4, 3), 60, dtype=np.uint8), rect_panel])

        # ── Status bar ───────────────────────────────────────────────────────
        n_border = sum(1 for tid in corners_dict if tid in tag_centres_mm)
        n_obj    = sum(1 for tid in corners_dict
                       if tid not in tag_centres_mm and tid in BORDERED_IDS)

        if calib['done']:
            status = "Calibration complete — see terminal for PERSP values.  C=redo  Q=quit"
            color  = (0, 220, 0)

        elif calib['active']:
            step = calib['step']
            corner_name = CALIB_CORNER_NAMES[step]
            true_x, true_y = corner_positions[step]
            has_tag = args.block_tag in corners_dict and H is not None
            obs_info = ""
            if has_tag:
                ox, oy = tag_interior_pos(
                    corners_dict[args.block_tag], H, scale, interior_inset_mm)
                obs_info = f"  obs=({ox:.1f},{oy:.1f})"
            status = (f"Corner {step+1}/4: place block at {corner_name} "
                      f"corner — tag centre→({true_x:.1f},{true_y:.1f})mm"
                      f"{obs_info}  SPACE=record  C=abort")
            color = (0, 220, 220) if has_tag else (0, 100, 220)

        else:
            status = (f"Border: {n_border}/{len(tag_centres_mm)}  "
                      f"Inliers: {n_inliers if H is not None else 0}  "
                      f"Objects: {n_obj}  "
                      f"{'[OK]' if H is not None else '[NO HOMOGRAPHY]'}  "
                      f"C=calibrate (4 corners)")
            color = (0, 220, 0) if H is not None else (0, 0, 220)

        cv2.putText(combined, status, (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        cv2.imshow("ArUco Homography 3D  |  Raw  |  Rectified", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('s'):
            fname = f"homography3d_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved {fname}")

        elif key == ord('c'):
            calib.update({
                'active': not calib['active'] or calib['done'],
                'step': 0, 'data': [], 'done_px': [], 'done': False,
                'corner_positions': corner_positions,
                'block_tag': args.block_tag,
                'block_w': args.block_w,
                'block_d': args.block_d,
            })
            if calib['active']:
                print(f"\n=== Calibration started ===")
                print(f"Place block at each of the 4 interior corners (TL→TR→BL→BR).")
                print(f"Block tag ID: {args.block_tag}  Height: {args.block_height}mm")
                print(f"Press SPACE to record each corner.  C to abort.")
            else:
                print("Calibration aborted/reset.")

        elif key == ord(' '):
            if not calib['active'] or calib['done']:
                continue
            step = calib['step']
            if H is None:
                print(f"  [{CALIB_CORNER_NAMES[step]}] No homography — need more border tags")
            elif args.block_tag not in corners_dict:
                print(f"  [{CALIB_CORNER_NAMES[step]}] Tag ID {args.block_tag} not detected")
            else:
                ox, oy = tag_interior_pos(
                    corners_dict[args.block_tag], H, scale, interior_inset_mm)
                true_x, true_y = corner_positions[step]
                h_mm = args.block_height
                calib['data'].append((true_x, true_y, ox, oy, h_mm))
                inset_px = int(interior_inset_mm * scale)
                calib['done_px'].append(
                    (inset_px + int(true_x * scale), inset_px + int(true_y * scale)))
                print(f"  [{CALIB_CORNER_NAMES[step]}] true=({true_x:.1f},{true_y:.1f})  "
                      f"obs=({ox:.2f},{oy:.2f})  err=({ox-true_x:+.2f},{oy-true_y:+.2f})")
                calib['step'] += 1

                if calib['step'] >= 4:
                    calib['active'] = False
                    calib['done']   = True
                    fit_linear_2d(calib['data'])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
