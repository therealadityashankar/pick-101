"""test_aruco_3d.py — 3D scene from flat ArUco board + perspective calibration.

Uses the flat board homography for XY position and applies perspective
correction (from camera_calibration.npz) for objects at known heights.

Panels:
    Left  — 3D perspective render (board, detected objects)
    Right — raw camera with tag outlines

Known-height tags: edit KNOWN_HEIGHT_TAGS to add your cube tag IDs.

Usage:
    python test_aruco_3d.py
    python test_aruco_3d.py --camera 1 --calibration camera_calibration.npz
    python test_aruco_3d.py --object-h 30 --calibration camera_calibration.npz
"""

import argparse
import time

import cv2
import cv2.aruco as aruco
import numpy as np


# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()

# ── Known-height object tags: {tag_id: height_above_board_mm} ────────────────
KNOWN_HEIGHT_TAGS = {
    120: 20.0,   # 20mm calibration cube
    130: 30.0,   # 30mm calibration cube
    150: 50.0,   # 50mm calibration cube
    101: 25.0,   # test tag at 25mm
    104: 0,   # test tag at 25mm — update height as needed
}

OUT_3D_W, OUT_3D_H = 640, 480


# ── Board geometry ────────────────────────────────────────────────────────────

def board_tag_centres(square_mm, tag_mm, gap_mm, n_outer, n_inner):
    """Return {tag_id: (x_mm, y_mm)} for all border tags. Origin = top-left of square."""
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


# ── Flat homography (image → board mm, Z=0) ───────────────────────────────────

def compute_flat_H(corners_dict, board_centres):
    """Homography mapping image pixels to board mm. Returns (H, n_inliers)."""
    src, dst = [], []
    for tid, ic in corners_dict.items():
        if tid not in board_centres:
            continue
        src.append([float(ic[:, 0].mean()), float(ic[:, 1].mean())])
        dst.append(list(board_centres[tid]))
    if len(src) < 4:
        return None, 0
    H, mask = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 5.0)
    return H, int(mask.sum()) if mask is not None else 0


def img_to_board_mm(pt, H):
    r = cv2.perspectiveTransform(np.float32([[[pt[0], pt[1]]]]), H)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


# ── Perspective calibration correction ───────────────────────────────────────

def apply_calib(obs_x, obs_y, h_mm, calib):
    """Apply perspective correction given calib=(Cx, Cy, Cz) or None."""
    if calib is None:
        return obs_x, obs_y
    Cx, Cy, Cz = calib
    if abs(Cz - h_mm) < 1e-6:
        return obs_x, obs_y
    scale = Cz / (Cz - h_mm)
    return Cx + (obs_x - Cx) / scale, Cy + (obs_y - Cy) / scale


# ── 3D virtual-camera rendering ───────────────────────────────────────────────

def _build_vcam(square_mm, out_w, out_h):
    """Fixed virtual camera looking at the board from above-front-right."""
    focus = np.array([square_mm * 0.5, square_mm * 0.55, 50.0])
    d     = square_mm * 1.65
    cam   = focus + np.array([d * 0.30, -d * 0.88, d * 0.78])

    fwd   = focus - cam;            fwd   /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0,0,1]); right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)

    Rv = np.array([right, -up, fwd])
    tv = -Rv @ cam
    f  = out_w * 0.62
    Kv = np.array([[f,0,out_w/2],[0,f,out_h/2],[0,0,1]], dtype=np.float64)
    return Kv, Rv, tv


def _vproj(pts_3d, Kv, Rv, tv):
    pts = np.atleast_2d(np.array(pts_3d, dtype=np.float64))
    pts = pts[:, [1, 0, 2]]   # swap X↔Y to match printed board orientation
    cam = (Rv @ pts.T).T + tv
    ok  = cam[:, 2] > 1.0
    px  = np.full((len(pts), 2), -1, dtype=int)
    if ok.any():
        z = cam[ok, 2]
        px[ok, 0] = (Kv[0,0] * cam[ok,0] / z + Kv[0,2]).astype(int)
        px[ok, 1] = (Kv[1,1] * cam[ok,1] / z + Kv[1,2]).astype(int)
    return px, ok


def render_3d_view(
    square_mm, inset_mm, isize_mm,
    board_centres, corners_dict,
    H_flat,
    out_w, out_h,
    calib=None,
):
    img  = np.full((out_h, out_w, 3), 18, dtype=np.uint8)
    Kv, Rv, tv = _build_vcam(square_mm, out_w, out_h)

    def proj(pts):
        return _vproj(pts, Kv, Rv, tv)

    def vline(a, b, col, th=1):
        px, v = proj([a, b])
        if v.all():
            cv2.line(img, tuple(px[0]), tuple(px[1]), col, th)

    def vquad(pts4, outline_col, fill_col=None, alpha=0.55):
        px, v = proj(pts4)
        if not v.all():
            return
        poly = px.reshape(-1, 1, 2)
        if fill_col is not None:
            ov = img.copy()
            cv2.fillPoly(ov, [poly], fill_col)
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [poly], True, outline_col, 1)

    ii   = inset_mm
    iz   = isize_mm
    s    = square_mm
    imax = ii + iz
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Board plane + interior square
    vquad([(0,0,0),(s,0,0),(s,s,0),(0,s,0)], (100,100,100), (35,35,35))
    vquad([(ii,ii,0),(ii+iz,ii,0),(ii+iz,ii+iz,0),(ii,ii+iz,0)],
          (150,150,150), (55,55,55))

    # Interior grid every 20mm
    for d in range(20, int(iz), 20):
        vline((ii+d, ii, 0), (ii+d, imax, 0), (65,65,65))
        vline((ii, ii+d, 0), (imax, ii+d, 0), (65,65,65))

    # Corner directional arrows
    ax = 18

    def varrow(a, b, col, th=1):
        pxa, va = proj([a]); pxb, vb = proj([b])
        if va[0] and vb[0]:
            cv2.arrowedLine(img, tuple(pxa[0]), tuple(pxb[0]), col, th, tipLength=0.25)

    def vlabel(pt, text, col):
        px, v = proj([pt])
        if v[0]:
            cv2.putText(img, text, (px[0][0]+3, px[0][1]-3), font, 0.22, col, 1)

    varrow((ii, ii, 0), (ii+ax, ii, 0), (80, 80, 220), 2)
    varrow((ii, ii, 0), (ii, ii+ax, 0), (80, 220, 80), 2)
    varrow((ii, ii, 0), (ii, ii, ax),   (80, 220, 220), 2)
    vlabel((ii+ax, ii, 0), "X", (80, 80, 220))
    vlabel((ii, ii+ax, 0), "Y", (80, 220, 80))
    vlabel((ii, ii, ax),   "Z", (80, 220, 220))
    # Dim arrows at other corners
    varrow((imax, ii,   0), (imax-ax, ii,   0), (60, 60, 160))
    varrow((imax, ii,   0), (imax,    ii+ax, 0), (60, 160, 60))
    varrow((ii,   imax, 0), (ii+ax,   imax,  0), (60, 60, 160))
    varrow((ii,   imax, 0), (ii,      imax-ax,0),(60, 160, 60))
    varrow((imax, imax, 0), (imax-ax, imax,  0), (60, 60, 160))
    varrow((imax, imax, 0), (imax,    imax-ax,0),(60, 160, 60))

    # Nadir marker from calibration
    if calib is not None:
        Cx, Cy, Cz = calib
        px, v = proj([(Cx, Cy, 0)])
        if v[0]:
            cv2.drawMarker(img, tuple(px[0]), (0, 200, 255),
                           cv2.MARKER_STAR, 12, 2)
            cv2.putText(img, f"nadir ({Cx:.0f},{Cy:.0f})",
                        (px[0][0]+6, px[0][1]), font, 0.26, (0,200,255), 1)

    # Board tags (green dots)
    for tid, ic in corners_dict.items():
        if tid not in board_centres:
            continue
        x, y = board_centres[tid]
        px, v = proj([(x, y, 0)])
        if v[0]:
            cv2.circle(img, tuple(px[0]), 2, (0, 200, 0), -1)

    # Object tags
    for tid, ic in corners_dict.items():
        if tid in board_centres or H_flat is None:
            continue
        cx = float(ic[:, 0].mean()); cy = float(ic[:, 1].mean())

        if tid in KNOWN_HEIGHT_TAGS:
            h_mm     = KNOWN_HEIGHT_TAGS[tid]
            dot_col  = (0, 200, 255)
            line_col = (0, 160, 200)
        else:
            h_mm     = 0.0
            dot_col  = (0, 80, 255)
            line_col = (0, 60, 200)

        bx, by = img_to_board_mm((cx, cy), H_flat)
        bx, by = apply_calib(bx, by, h_mm, calib)
        pt3d   = np.array([bx, by, h_mm])

        pf, vf = proj([(pt3d[0], pt3d[1], 0)])
        po, vo = proj([(pt3d[0], pt3d[1], pt3d[2])])
        if vf[0] and vo[0]:
            cv2.line(img, tuple(pf[0]), tuple(po[0]), line_col, 1)
        if vo[0]:
            cv2.circle(img, tuple(po[0]), 6, dot_col, -1)
            # Show interior coords (subtract inset)
            ix = pt3d[0] - inset_mm
            iy = pt3d[1] - inset_mm
            lbl = f"ID{tid}  h={h_mm:.0f}mm  int=({ix:.0f},{iy:.0f})"
            cv2.putText(img, lbl, (po[0][0]+5, po[0][1]-3), font, 0.28, dot_col, 1)

    # Calibration info
    if calib is not None:
        Cx, Cy, Cz = calib
        cv2.putText(img, f"calib: nadir=({Cx:.0f},{Cy:.0f}) Cz={Cz:.0f}mm",
                    (6, out_h-5), font, 0.30, (130,130,130), 1)
    else:
        cv2.putText(img, "no calibration — heights uncorrected",
                    (6, out_h-5), font, 0.30, (80,80,80), 1)

    return img


# ── Camera overlay panel ──────────────────────────────────────────────────────

def draw_cam_panel(frame, corners_dict, board_centres, H_flat, square_mm):
    out  = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tid, corners in corners_dict.items():
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        if tid in board_centres:
            col = (0, 220, 0)
        elif tid in KNOWN_HEIGHT_TAGS:
            col = (0, 200, 255)
        else:
            col = (0, 80, 255)
        cv2.polylines(out, [pts], True, col, 2)
        cx = int(corners[:, 0].mean()); cy = int(corners[:, 1].mean())
        if tid in KNOWN_HEIGHT_TAGS:
            lbl = f"{tid}({KNOWN_HEIGHT_TAGS[tid]:.0f}mm)"
        else:
            lbl = str(tid)
        cv2.putText(out, lbl, (cx-8, cy+5), font, 0.30, col, 1, cv2.LINE_AA)

    if H_flat is not None:
        Hi    = np.linalg.inv(H_flat)
        brd   = np.float32([[0,0],[square_mm,0],[square_mm,square_mm],[0,square_mm]])
        img_c = cv2.perspectiveTransform(brd.reshape(1,-1,2), Hi).reshape(-1, 2)
        cv2.polylines(out, [img_c.astype(np.int32).reshape(-1,1,2)],
                      True, (0,200,255), 2)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",      type=int,   default=0)
    ap.add_argument("--square-mm",   type=float, default=180.0)
    ap.add_argument("--tag-mm",      type=float, default=16.0)
    ap.add_argument("--gap-mm",      type=float, default=2.0)
    ap.add_argument("--n-outer",     type=int,   default=10)
    ap.add_argument("--n-inner",     type=int,   default=8)
    ap.add_argument("--calibration", type=str,   default=None,
                    help="Path to camera_calibration.npz from calibrate_3d.py")
    args = ap.parse_args()

    board_centres = board_tag_centres(
        args.square_mm, args.tag_mm, args.gap_mm, args.n_outer, args.n_inner)
    inset_mm = 2 * args.tag_mm + args.gap_mm
    isize_mm = args.square_mm - 2 * inset_mm

    print(f"Board:    {args.square_mm:.0f}mm  |  {len(board_centres)} border tags")
    print(f"Interior: {isize_mm:.1f}×{isize_mm:.1f} mm  (inset {inset_mm:.1f} mm)")
    print("Press Q to quit, S to save snapshot")

    calib = None
    if args.calibration:
        try:
            d     = np.load(args.calibration)
            calib = (float(d['Cx']), float(d['Cy']), float(d['Cz']))
            print(f"Calibration: nadir=({calib[0]:.1f},{calib[1]:.1f}) Cz={calib[2]:.1f}mm")
        except Exception as e:
            print(f"Warning: could not load calibration: {e}")

    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    cap      = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    H_flat = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dc, di, _ = detector.detectMarkers(gray)

        corners_dict = {}
        if di is not None:
            for i, tid in enumerate(di.flatten()):
                corners_dict[int(tid)] = dc[i].reshape(4, 2)

        H_flat, n_inliers = compute_flat_H(corners_dict, board_centres)

        view3d = render_3d_view(
            args.square_mm, inset_mm, isize_mm,
            board_centres, corners_dict,
            H_flat,
            OUT_3D_W, OUT_3D_H,
            calib=calib,
        )
        campanel = draw_cam_panel(
            frame, corners_dict, board_centres, H_flat, args.square_mm)

        h  = max(view3d.shape[0], campanel.shape[0])
        v3 = cv2.resize(view3d,   (int(view3d.shape[1] * h / view3d.shape[0]), h))
        cp = cv2.resize(campanel, (int(campanel.shape[1]* h / campanel.shape[0]), h))
        combined = np.hstack([v3, np.full((h, 4, 3), 50, np.uint8), cp])

        n_board = sum(1 for tid in corners_dict if tid in board_centres)
        status  = (f"Board:{n_board}/{len(board_centres)}  "
                   f"Inliers:{n_inliers}  "
                   f"{'[HOMOGRAPHY OK]' if H_flat is not None else '[NEED MORE TAGS]'}  "
                   f"{'[CALIBRATED]' if calib else '[uncalibrated]'}")
        cv2.putText(combined, status, (8, combined.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0,220,0) if H_flat is not None else (120,120,120), 1, cv2.LINE_AA)

        cv2.imshow("ArUco 3D  |  3D view (left)  |  Camera (right)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"aruco3d_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
