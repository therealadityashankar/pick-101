"""calibrate_3d.py — 3D perspective calibration using a single jenga block at 4 corners.

Place the block so its top-left corner touches each interior corner of the board.
The script detects the ArUco tag, measures where it appears via flat homography,
compares to the known true position, and fits (Cx, Cy, Cz).

Correction model (exact pinhole):
    true_x = Cx + (obs_x - Cx) * (Cz - h) / Cz
    true_y = Cy + (obs_y - Cy) * (Cz - h) / Cz

Output: camera_calibration.npz  (Cx, Cy, Cz in interior mm)

Usage:
    uv run python calibrate_3d.py --port 0
    uv run python calibrate_3d.py --camera 1 --block-tag 101 --block-height 15 \\
        --block-w 25 --block-d 25 --tag-ox 12.5 --tag-oy 12.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.optimize import least_squares

# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.errorCorrectionRate       = 0.7

CAPTURE_FRAMES = 40   # frames averaged per measurement


# ── Board helpers (identical to visualize_irl_block.py) ───────────────────────

def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    centres = {}
    tag_id  = 0

    def side_centres(inset_mm, n, side):
        half   = tag_mm / 2
        margin = inset_mm + half
        span   = square_mm - 2 * margin
        out    = []
        for i in range(n):
            t     = i / (n - 1) if n > 1 else 0.5
            along = margin + t * span
            if   side == "top":    out.append((along, inset_mm + half))
            elif side == "bottom": out.append((along, square_mm - inset_mm - half))
            elif side == "left":
                if 0 < i < n - 1: out.append((inset_mm + half, along))
            elif side == "right":
                if 0 < i < n - 1: out.append((square_mm - inset_mm - half, along))
        return out

    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(0, n_outer, side):
            centres[tag_id] = (cx, cy); tag_id += 1
    inner_inset = tag_mm + gap_mm
    for side in ("top", "bottom", "left", "right"):
        for cx, cy in side_centres(inner_inset, n_inner, side):
            centres[tag_id] = (cx, cy); tag_id += 1
    return centres


def compute_homography(corners_dict, board_centres):
    src, dst = [], []
    for tid, ic in corners_dict.items():
        if tid not in board_centres:
            continue
        src.append([float(ic[:, 0].mean()), float(ic[:, 1].mean())])
        dst.append(list(board_centres[tid]))
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 5.0)
    return H


def img_to_board(cx, cy, H):
    r = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


# ── Calibration positions ─────────────────────────────────────────────────────

def make_corners(interior_mm, block_w, block_d):
    """Return list of (label, block_tl_x, block_tl_y) in interior mm.

    The block top-left corner is placed against each interior corner.
    The two right/bottom positions are inset by block_w / block_d so the
    block stays inside the interior region.
    """
    W, H = interior_mm, interior_mm
    return [
        ("TOP-LEFT corner",     0,          0),
        ("TOP-RIGHT corner",    W - block_w, 0),
        ("BOTTOM-LEFT corner",  0,           H - block_d),
        ("BOTTOM-RIGHT corner", W - block_w, H - block_d),
    ]


# ── Perspective fitting ───────────────────────────────────────────────────────

def fit_camera(measurements):
    """Fit (Cx, Cy, Cz) from (true_x, true_y, obs_x, obs_y, height_mm) list."""
    def residuals(params):
        Cx, Cy, Cz = params
        res = []
        for tx, ty, ox, oy, h in measurements:
            if abs(Cz - h) < 1e-6:
                res += [1e6, 1e6]; continue
            pred_ox = (Cz * tx - Cx * h) / (Cz - h)
            pred_oy = (Cz * ty - Cy * h) / (Cz - h)
            res += [ox - pred_ox, oy - pred_oy]
        return res

    x0     = [56.0, 56.0, 300.0]
    result = least_squares(residuals, x0,
                           bounds=([-300, -300, 50], [600, 600, 2000]),
                           method='trf')
    return result.x, result


def correct(obs_x, obs_y, h, Cx, Cy, Cz):
    scale = Cz / (Cz - h)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


# ── Capture helpers ───────────────────────────────────────────────────────────

def detect_frame(cap, detector, board_centres):
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dc, di, _ = detector.detectMarkers(gray)
    corners_dict = {}
    if di is not None:
        corners_dict = {int(t): dc[i].reshape(4, 2) for i, t in enumerate(di.flatten())}
    H = compute_homography(corners_dict, board_centres)
    return frame, corners_dict, H


def capture_avg(cap, detector, board_centres, tag_id, n=CAPTURE_FRAMES):
    """Average tag centre in board mm over n frames. Returns (bx, by) or None."""
    samples = []
    while len(samples) < n:
        _, corners_dict, H = detect_frame(cap, detector, board_centres)
        if H is None or tag_id not in corners_dict:
            continue
        ic = corners_dict[tag_id]
        cx, cy = float(ic[:, 0].mean()), float(ic[:, 1].mean())
        samples.append(img_to_board(cx, cy, H))
    return float(np.mean([s[0] for s in samples])), float(np.mean([s[1] for s in samples]))


def draw_ui(frame, corners_dict, board_centres, tag_id, H, inset_mm,
            label, step, total, obs_board=None):
    out  = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = out.shape[:2]

    # Highlight border tags and block tag
    for tid, ic in corners_dict.items():
        pts = ic.astype(np.int32).reshape(-1, 1, 2)
        if tid in board_centres:
            cv2.polylines(out, [pts], True, (0, 200, 0), 1)
        elif tid == tag_id:
            cv2.polylines(out, [pts], True, (0, 200, 255), 2)

    # Status bar
    cv2.rectangle(out, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.putText(out, f"Step {step}/{total} — Place block at {label}",
                (8, h - 56), font, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    if obs_board is not None:
        ox_int = obs_board[0] - inset_mm
        oy_int = obs_board[1] - inset_mm
        cv2.putText(out, f"Tag seen at interior ({ox_int:.1f}, {oy_int:.1f}) mm",
                    (8, h - 34), font, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
    elif H is None:
        cv2.putText(out, "Board not detected — need more tags visible",
                    (8, h - 34), font, 0.45, (0, 80, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"Block tag ID {tag_id} not detected",
                    (8, h - 34), font, 0.45, (80, 80, 255), 1, cv2.LINE_AA)

    cv2.putText(out, "SPACE = capture    S = skip    Q = quit",
                (8, h - 10), font, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",      type=int,   default=0)
    ap.add_argument("--square-mm",   type=float, default=180.0,
                    help="Board outer size in mm")
    ap.add_argument("--tag-mm",      type=float, default=16.0)
    ap.add_argument("--gap-mm",      type=float, default=2.0)
    ap.add_argument("--n-outer",     type=int,   default=10)
    ap.add_argument("--n-inner",     type=int,   default=8)
    ap.add_argument("--block-tag",   type=int,   default=101,
                    help="ArUco tag ID on the jenga block")
    ap.add_argument("--block-height",type=float, default=15.0,
                    help="Height of block (mm) — how high the tag sits above the board")
    ap.add_argument("--block-w",     type=float, default=25.0,
                    help="Block footprint width (mm), along interior X")
    ap.add_argument("--block-d",     type=float, default=25.0,
                    help="Block footprint depth (mm), along interior Y")
    ap.add_argument("--tag-ox",      type=float, default=None,
                    help="Tag centre X offset from block TL corner (mm). "
                         "Default: block-w / 2")
    ap.add_argument("--tag-oy",      type=float, default=None,
                    help="Tag centre Y offset from block TL corner (mm). "
                         "Default: block-d / 2")
    ap.add_argument("--out",         type=str,   default="camera_calibration.npz")
    args = ap.parse_args()

    tag_ox = args.tag_ox if args.tag_ox is not None else args.block_w / 2
    tag_oy = args.tag_oy if args.tag_oy is not None else args.block_d / 2

    inset_mm     = 2 * args.tag_mm + args.gap_mm
    interior_mm  = args.square_mm - 2 * inset_mm

    board_centres = board_tag_centres(
        args.square_mm, args.tag_mm, args.gap_mm, args.n_outer, args.n_inner)
    corner_steps  = make_corners(interior_mm, args.block_w, args.block_d)

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    cap      = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}"); sys.exit(1)

    print(f"\nBlock tag ID : {args.block_tag}")
    print(f"Block height : {args.block_height} mm")
    print(f"Block size   : {args.block_w} × {args.block_d} mm")
    print(f"Tag offset   : ({tag_ox:.1f}, {tag_oy:.1f}) mm from block TL corner")
    print(f"Interior     : {interior_mm:.1f} × {interior_mm:.1f} mm")
    print(f"\nPress C to start calibration, Q to quit.\n")

    # ── Wait for board ────────────────────────────────────────────────────────
    while True:
        frame, corners_dict, H = detect_frame(cap, detector, board_centres)
        if frame is None: break

        n_board = sum(1 for t in corners_dict if t in board_centres)
        status  = (f"Board tags: {n_board}/{len(board_centres)}  "
                   f"{'[HOMOGRAPHY OK]' if H is not None else '[NEED MORE TAGS]'}  "
                   f"| C = start  Q = quit")
        out = frame.copy()
        for tid, ic in corners_dict.items():
            col = (0, 200, 0) if tid in board_centres else (0, 80, 255)
            cv2.polylines(out, [ic.astype(np.int32).reshape(-1,1,2)], True, col, 1)
        cv2.rectangle(out, (0, out.shape[0]-24), (out.shape[1], out.shape[0]), (20,20,20), -1)
        cv2.putText(out, status, (8, out.shape[0]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0,220,0) if H is not None else (120,120,120), 1)
        cv2.imshow("Calibration", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): cap.release(); cv2.destroyAllWindows(); return
        if key == ord('c') and H is not None: break

    # ── Collect measurements ──────────────────────────────────────────────────
    measurements = []
    total = len(corner_steps)

    for step, (label, tl_x, tl_y) in enumerate(corner_steps, 1):
        # True tag centre in board coords (interior → board by adding inset)
        true_x = inset_mm + tl_x + tag_ox
        true_y = inset_mm + tl_y + tag_oy

        print(f"\n[{step}/{total}] {label}")
        print(f"  Block TL at interior ({tl_x:.0f}, {tl_y:.0f}) mm  "
              f"→  true tag centre at board ({true_x:.1f}, {true_y:.1f}) mm")
        print("  SPACE = capture   S = skip   Q = quit")

        action = None
        while action is None:
            frame, corners_dict, H = detect_frame(cap, detector, board_centres)
            if frame is None: break

            obs_board = None
            if H is not None and args.block_tag in corners_dict:
                ic = corners_dict[args.block_tag]
                cx, cy = float(ic[:, 0].mean()), float(ic[:, 1].mean())
                obs_board = img_to_board(cx, cy, H)

            out = draw_ui(frame, corners_dict, board_centres, args.block_tag,
                          H, inset_mm, label, step, total, obs_board)
            cv2.imshow("Calibration", out)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and obs_board is not None:
                action = 'capture'
            elif key == ord('s'):
                action = 'skip'
            elif key == ord('q'):
                action = 'quit'

        if action == 'quit':
            print("Aborted."); break
        if action == 'skip':
            print("  Skipped."); continue

        print(f"  Capturing {CAPTURE_FRAMES} frames — hold still...", end=" ", flush=True)
        # Show hold-still feedback
        cv2.imshow("Calibration", draw_ui(frame, {}, board_centres, args.block_tag,
                                           None, inset_mm, "HOLD STILL...", step, total))
        cv2.waitKey(1)

        obs_x, obs_y = capture_avg(cap, detector, board_centres, args.block_tag)
        obs_ix = obs_x - inset_mm
        obs_iy = obs_y - inset_mm
        err_x  = obs_x - true_x
        err_y  = obs_y - true_y
        print(f"OK  obs interior=({obs_ix:.1f}, {obs_iy:.1f})  "
              f"flat err=({err_x:+.1f}, {err_y:+.1f}) mm")

        measurements.append((true_x, true_y, obs_x, obs_y, args.block_height))

    cap.release()
    cv2.destroyAllWindows()

    # ── Fit ───────────────────────────────────────────────────────────────────
    if len(measurements) < 3:
        print(f"\nOnly {len(measurements)} measurement(s) — need at least 3. Aborting.")
        sys.exit(1)

    print(f"\nFitting to {len(measurements)} measurements...")
    (Cx, Cy, Cz), result = fit_camera(measurements)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Camera nadir  : ({Cx:.2f}, {Cy:.2f}) mm in board coords")
    print(f"Camera height : {Cz:.2f} mm above board")
    print(f"{'='*60}")
    print(f"\n{'Label':20s}  {'True (int)':>16}  {'Obs (int)':>16}  "
          f"{'Flat err':>10}  {'Corr err':>10}")
    print("-"*80)

    labels = [l for l, *_ in corner_steps]
    raw_errs, corr_errs = [], []
    for i, (tx, ty, ox, oy, h) in enumerate(measurements):
        cx, cy = correct(ox, oy, h, Cx, Cy, Cz)
        raw_e  = np.hypot(ox - tx, oy - ty)
        corr_e = np.hypot(cx - tx, cy - ty)
        raw_errs.append(raw_e);  corr_errs.append(corr_e)
        lbl = labels[i] if i < len(labels) else f"pos {i}"
        print(f"{lbl:20s}  ({tx-inset_mm:6.1f},{ty-inset_mm:6.1f})  "
              f"({ox-inset_mm:6.1f},{oy-inset_mm:6.1f})  "
              f"{raw_e:>10.2f}mm  {corr_e:>10.2f}mm")

    print("-"*80)
    print(f"{'Mean flat error:':30s} {np.mean(raw_errs):.2f} mm")
    print(f"{'Mean corrected error:':30s} {np.mean(corr_errs):.2f} mm")
    print(f"{'Max  corrected error:':30s} {np.max(corr_errs):.2f} mm")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    np.savez(out_path,
             Cx=Cx, Cy=Cy, Cz=Cz,
             inset_mm=inset_mm,
             square_mm=args.square_mm,
             measurements=np.array(measurements))
    print(f"\nSaved → {out_path}")
    print(f"\nPaste these into visualize_irl_block.py / run_real_t1.py:")
    print(f"  PERSP_Cx = {Cx:.2f}")
    print(f"  PERSP_Cy = {Cy:.2f}")
    print(f"  PERSP_Cz = {Cz:.2f}")


if __name__ == "__main__":
    main()
