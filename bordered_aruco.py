"""bordered_aruco.py — Bordered ArUco tag definition and renderer.

A bordered ArUco tag has the following structure (cross-section, top to bottom):

    [outer border]
    [double-border gap]      ← only visible as extra whitespace on right/bottom
    [double border line]     ← drawn only on TOP and LEFT sides (L-shape indicator)
    [quiet zone]
    [ArUco marker bits]
    [quiet zone]
    [double border line]     ← not drawn (just whitespace) on right/bottom
    [double-border gap]
    [outer border]

The double-border L at the top-left corner unambiguously identifies orientation.
The image is square and the ArUco marker is centred within it.

All four margins are equal (= border + gap + inner + quiet_zone), so the
right and bottom sides have a wider effective quiet zone — that is fine, ArUco
only needs at least one cell of quiet zone.

Usage:
    python bordered_aruco.py                  # preview ID 0 at 400px
    python bordered_aruco.py --id 42          # preview ID 42
    python bordered_aruco.py --id 5 --size 200 --save out.png
"""

import argparse

import cv2
import cv2.aruco as aruco
import numpy as np


ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


def generate_bordered_tag(
    tag_id: int,
    tag_size_px: int = 200,
    quiet_zone_px: int = 24,
    border_px: int = 10,
    double_gap_px: int = 8,
    double_border_px: int = 6,
    double_length_ratio: float = 0.45,
    aruco_dict=None,
) -> np.ndarray:
    """Return a grayscale image of a bordered ArUco tag.

    Parameters
    ----------
    tag_id : int
        ArUco marker ID (0–249 for DICT_6X6_250).
    tag_size_px : int
        Pixel size of the ArUco marker itself (the bit grid).
    quiet_zone_px : int
        White space between the innermost border line and the ArUco bits.
    border_px : int
        Thickness of the outer border rectangle (drawn on all four sides).
    double_gap_px : int
        Gap between the outer border and the inner indicator line.
    double_border_px : int
        Thickness of the inner indicator line drawn at the top-left corner.
    double_length_ratio : float
        Fraction (0–1) of each side length covered by the indicator line.
    aruco_dict :
        ArUco dictionary. Defaults to DICT_6X6_250.

    Returns
    -------
    np.ndarray, dtype uint8, shape (H, W) — grayscale image.
    """
    if aruco_dict is None:
        aruco_dict = ARUCO_DICT

    # Generate raw ArUco marker (black border included)
    tag_img = np.zeros((tag_size_px, tag_size_px), dtype=np.uint8)
    aruco.generateImageMarker(aruco_dict, tag_id % 250, tag_size_px, tag_img, 1)

    # Total margin on every side (ensures the image is square and tag is centred)
    margin = border_px + double_gap_px + double_border_px + quiet_zone_px
    total = tag_size_px + 2 * margin

    # White background
    img = np.ones((total, total), dtype=np.uint8) * 255

    # Place ArUco marker
    img[margin : margin + tag_size_px, margin : margin + tag_size_px] = tag_img

    # ── Outer border (all four sides) ────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (total - 1, total - 1), 0, border_px)

    # ── Double-border indicator at top-left (L-shape) ─────────────────────────
    # Position of the inner line (measured from the image edge)
    inner_pos = border_px + double_gap_px + double_border_px // 2

    # How long the indicator arms are
    arm_len = int(total * double_length_ratio)

    # Horizontal arm: runs along the top, from left edge to arm_len
    cv2.line(img, (0, inner_pos), (arm_len, inner_pos), 0, double_border_px)

    # Vertical arm: runs along the left, from top edge to arm_len
    cv2.line(img, (inner_pos, 0), (inner_pos, arm_len), 0, double_border_px)

    return img


def tag_physical_size(
    tag_size_mm: float,
    quiet_zone_frac: float = 0.12,
    border_frac: float = 0.05,
    double_gap_frac: float = 0.04,
    double_border_frac: float = 0.03,
) -> dict:
    """Return physical dimensions (in mm) for a bordered tag printed at tag_size_mm.

    tag_size_mm is the physical width of the ArUco bit grid.
    All other dimensions are proportional to tag_size_mm by default.

    Returns a dict with keys:
      tag_mm, quiet_zone_mm, border_mm, double_gap_mm, double_border_mm,
      total_mm (total physical size of the printed square)
    """
    quiet = tag_size_mm * quiet_zone_frac
    border = tag_size_mm * border_frac
    dgap = tag_size_mm * double_gap_frac
    dborder = tag_size_mm * double_border_frac
    margin = border + dgap + dborder + quiet
    total = tag_size_mm + 2 * margin
    return dict(
        tag_mm=tag_size_mm,
        quiet_zone_mm=quiet,
        border_mm=border,
        double_gap_mm=dgap,
        double_border_mm=dborder,
        margin_mm=margin,
        total_mm=total,
    )


def main():
    ap = argparse.ArgumentParser(description="Preview / export a bordered ArUco tag.")
    ap.add_argument("--id",    type=int, default=0,    help="ArUco ID (default: 0)")
    ap.add_argument("--size",  type=int, default=400,  help="Tag bit-grid size in pixels (default: 400)")
    ap.add_argument("--save",  type=str, default=None, help="Save image to this path instead of displaying")
    ap.add_argument("--ratio", type=float, default=0.45, help="Indicator arm length ratio (default: 0.45)")
    args = ap.parse_args()

    img = generate_bordered_tag(
        tag_id=args.id,
        tag_size_px=args.size,
        double_length_ratio=args.ratio,
    )

    # Convert to BGR for display / save
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Overlay label
    label = f"Bordered ArUco ID={args.id}  (DICT_6X6_250)"
    cv2.putText(bgr, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)

    info = (f"tag={args.size}px  total={img.shape[0]}px  "
            f"top-left L-indicator shows orientation")
    cv2.putText(bgr, info, (8, bgr.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    if args.save:
        cv2.imwrite(args.save, bgr)
        print(f"Saved {args.save}")
    else:
        cv2.imshow(f"Bordered ArUco ID={args.id}", bgr)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
