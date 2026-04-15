"""Live ArUco homography test.

Opens the camera, detects border tags on the printed board, computes a
homography from the tag layout to a canonical flat-board view, and shows:
  Left  — raw camera feed with detected tags highlighted
  Right — rectified bird's-eye view of the board interior

The board's interior corners (where the ruler grid starts) are mapped to
the right panel so you can drop any tag inside the square and see its
corrected position.

Usage:
    python test_aruco_homography.py
    python test_aruco_homography.py --camera 1 --square-mm 180 --tag-mm 16
"""
import argparse
import cv2
import cv2.aruco as aruco
import numpy as np

# ── Board geometry (must match make_aruco_board.py defaults) ─────────────────
# IDs 0..n_outer*4-1  → outer row
# IDs n_outer*4..end  → inner row
# We use ALL detected border tags for the homography — more tags = more robust.

ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()

# Output panel size for the rectified view
OUT_W = 600
OUT_H = 600


def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    """Return {tag_id: (x_mm, y_mm)} for all border tags.

    Origin = top-left corner of the square.
    X → right, Y → down.
    """
    centres = {}
    tag_id = 0

    def side_centres(inset_mm, n, side):
        half = tag_mm / 2
        margin = inset_mm + half
        span = square_mm - 2 * margin
        positions = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
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


def compute_homography(corners_dict, tag_centres_mm, square_mm, tag_mm, gap_mm):
    """Build src (image pixels) → dst (mm on board) point correspondences.

    Uses the centre of each detected tag.
    Returns H (3×3) or None if too few matches.
    """
    src_pts = []
    dst_pts = []

    for tag_id, img_corners in corners_dict.items():
        if tag_id not in tag_centres_mm:
            continue
        # Image centre of tag = mean of 4 corners
        cx_img = float(img_corners[:, 0].mean())
        cy_img = float(img_corners[:, 1].mean())
        cx_mm, cy_mm = tag_centres_mm[tag_id]
        src_pts.append([cx_img, cy_img])
        dst_pts.append([cx_mm, cy_mm])

    if len(src_pts) < 4:
        return None, len(src_pts)

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    # Scale mm → output pixels
    scale = OUT_W / square_mm
    dst_px = dst * scale

    H, mask = cv2.findHomography(src, dst_px, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def draw_raw_panel(frame, corners_dict, tag_centres_mm, H, square_mm):
    """Draw detected tags and reprojection onto the raw camera frame."""
    out = frame.copy()

    for tag_id, corners in corners_dict.items():
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        known = tag_id in tag_centres_mm
        color = (0, 220, 0) if known else (0, 100, 255)
        cv2.polylines(out, [pts], True, color, 2)
        cx = int(corners[:, 0].mean())
        cy = int(corners[:, 1].mean())
        cv2.putText(out, str(tag_id), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Reproject board corners back into image via H inverse
    if H is not None:
        H_inv = np.linalg.inv(H)
        scale = OUT_W / square_mm
        board_corners_px = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]],
                                     dtype=np.float32) / scale * square_mm
        board_corners_px *= scale  # already in output px, need mm first
        # corners in mm
        brd = np.array([[0, 0], [square_mm, 0], [square_mm, square_mm], [0, square_mm]],
                        dtype=np.float32)
        brd_out_px = brd * scale
        # back-project via H_inv
        brd_img = cv2.perspectiveTransform(brd_out_px.reshape(1, -1, 2), H_inv).reshape(-1, 2)
        cv2.polylines(out, [brd_img.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 2)

    return out


def draw_rectified_panel(frame, H, square_mm, tag_mm, gap_mm, corners_dict,
                          tag_centres_mm, inner_tags_start):
    """Warp frame to bird's-eye board view and overlay grid + any detected inner tags.

    Measurements are shown relative to the interior origin (inside the black border lines),
    matching the ruler printed on the board.
    """
    if H is None:
        blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(blank, "Need >= 4 border tags", (20, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return blank

    warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))
    scale = OUT_W / square_mm

    # Interior region in mm (matches make_aruco_board.py: 2*tag + gap from each edge)
    interior_inset_mm = 2 * tag_mm + gap_mm
    interior_size_mm  = square_mm - 2 * interior_inset_mm
    inset_px = int(interior_inset_mm * scale)
    isize_px = int(interior_size_mm * scale)

    # Draw outer square border
    cv2.rectangle(warped, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)

    # Draw thick black interior border (matching the printed board)
    cv2.rectangle(warped,
                  (inset_px, inset_px),
                  (inset_px + isize_px, inset_px + isize_px),
                  (0, 0, 0), 2)

    # Grid lines clipped to interior, relative to interior origin
    for d_mm in range(5, int(interior_size_mm), 5):
        d_px = inset_px + int(d_mm * scale)
        is_major = d_mm % 10 == 0
        color = (100, 100, 100) if is_major else (180, 180, 180)
        thick = 1
        # clip to interior
        cv2.line(warped, (d_px, inset_px), (d_px, inset_px + isize_px), color, thick)
        cv2.line(warped, (inset_px, d_px), (inset_px + isize_px, d_px), color, thick)
        if is_major:
            label = str(d_mm)
            cv2.putText(warped, label, (d_px + 2, inset_px + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(warped, label, (inset_px + 2, d_px - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

    # Highlight border tag positions (known)
    for tag_id, (cx_mm, cy_mm) in tag_centres_mm.items():
        cx_px = int(cx_mm * scale)
        cy_px = int(cy_mm * scale)
        color = (0, 180, 0) if tag_id < inner_tags_start else (0, 120, 200)
        cv2.circle(warped, (cx_px, cy_px), 4, color, -1)

    # Draw any detected tags that are NOT border tags (objects placed inside)
    for tag_id, corners in corners_dict.items():
        if tag_id in tag_centres_mm:
            continue  # border tag, skip
        # Warp corners into board space
        warped_corners = cv2.perspectiveTransform(
            corners.reshape(1, -1, 2), H).reshape(-1, 2)
        cx_px = int(warped_corners[:, 0].mean())
        cy_px = int(warped_corners[:, 1].mean())
        # Position relative to interior origin
        cx_interior_mm = (cx_px / scale) - interior_inset_mm
        cy_interior_mm = (cy_px / scale) - interior_inset_mm
        pts = warped_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(warped, [pts], True, (0, 0, 255), 2)
        cv2.circle(warped, (cx_px, cy_px), 5, (0, 0, 255), -1)
        cv2.putText(warped, f"ID{tag_id}  x={cx_interior_mm:.1f} y={cy_interior_mm:.1f} mm",
                    (cx_px + 6, cy_px - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255), 1, cv2.LINE_AA)

    return warped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--square-mm", type=float, default=180.0)
    parser.add_argument("--tag-mm",    type=float, default=16.0)
    parser.add_argument("--gap-mm",    type=float, default=2.0)
    parser.add_argument("--n-outer",   type=int,   default=10,
                        help="Tags per side in outer row (default: 10)")
    parser.add_argument("--n-inner",   type=int,   default=8,
                        help="Tags per side in inner row (default: 8)")
    args = parser.parse_args()

    tag_centres_mm = board_tag_centres(
        args.square_mm, args.tag_mm, args.gap_mm, args.n_outer, args.n_inner)
    inner_tags_start = args.n_outer * 4 - (args.n_outer - 2) * 2  # approx outer count
    # Exact: outer row places n_outer top/bottom + (n_outer-2) left/right each
    n_outer_total = args.n_outer * 2 + (args.n_outer - 2) * 2
    inner_tags_start = n_outer_total

    print(f"Board: {args.square_mm:.0f}mm  |  {len(tag_centres_mm)} border tags  "
          f"(IDs 0–{len(tag_centres_mm)-1})")
    print(f"Outer row: IDs 0–{inner_tags_start-1}  |  Inner row: IDs {inner_tags_start}–{len(tag_centres_mm)-1}")
    print("Press Q to quit, S to save snapshot")

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        return

    H = None

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

        # Recompute homography every frame
        H, n_inliers = compute_homography(
            corners_dict, tag_centres_mm, args.square_mm, args.tag_mm, args.gap_mm)

        raw_panel = draw_raw_panel(frame, corners_dict, tag_centres_mm, H, args.square_mm)
        rect_panel = draw_rectified_panel(
            frame, H, args.square_mm, args.tag_mm, args.gap_mm,
            corners_dict, tag_centres_mm, inner_tags_start)

        # Resize both panels to same height
        h = max(raw_panel.shape[0], rect_panel.shape[0])
        raw_panel  = cv2.resize(raw_panel,  (int(raw_panel.shape[1]  * h / raw_panel.shape[0]),  h))
        rect_panel = cv2.resize(rect_panel, (int(rect_panel.shape[1] * h / rect_panel.shape[0]), h))

        combined = np.hstack([raw_panel, np.full((h, 4, 3), 60, dtype=np.uint8), rect_panel])

        # Status bar
        n_border = sum(1 for tid in corners_dict if tid in tag_centres_mm)
        n_inner  = sum(1 for tid in corners_dict if tid not in tag_centres_mm)
        status = (f"Border tags: {n_border}/{len(tag_centres_mm)}  "
                  f"Inliers: {n_inliers if H is not None else 0}  "
                  f"Inner objects: {n_inner}  "
                  f"{'[HOMOGRAPHY OK]' if H is not None else '[NO HOMOGRAPHY]'}")
        cv2.putText(combined, status, (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 0) if H is not None else (0, 0, 220), 1, cv2.LINE_AA)

        cv2.imshow("ArUco Homography Test  |  Raw  |  Rectified", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import time
            fname = f"homography_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
