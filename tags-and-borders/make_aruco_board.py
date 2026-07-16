"""Generate a printable A4 PDF with a large square bordered by two rows of ArUco tags.

The square is centred on the page. ArUco tags line the perimeter in two concentric
rows so the homography can be robustly estimated even if some tags are occluded.

Output: aruco_board.pdf  (also saves aruco_board.png for preview)

Usage:
    python make_aruco_board.py
    python make_aruco_board.py --square-mm 180 --tag-mm 18 --out my_board.pdf
"""
import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas
from PIL import Image
import io


# ── Constants ────────────────────────────────────────────────────────────────

# ArUco dictionary — 6x6 gives 250 unique IDs, plenty for two border rows
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# DPI for rasterising the board image before embedding in PDF
DPI = 300


def mm_to_px(mm_val: float, dpi: int = DPI) -> int:
    return int(round(mm_val / 25.4 * dpi))


def px_to_mm(px: int, dpi: int = DPI) -> float:
    return px / dpi * 25.4


def draw_board(
    square_mm: float,
    tag_mm: float,
    gap_mm: float,
    dpi: int = DPI,
) -> np.ndarray:
    """Render the board as a grayscale image and return it.

    Layout:
        - Outer row of tags: flush against the square border
        - Inner row: one tag-width + gap inward
        - Tags are evenly spaced along each side
        - Corner positions are shared between adjacent sides

    Returns an RGBA numpy array (H, W, 4).
    """
    sq_px   = mm_to_px(square_mm, dpi)
    tag_px  = mm_to_px(tag_mm, dpi)
    gap_px  = mm_to_px(gap_mm, dpi)
    row2_inset_px = tag_px + gap_px  # inset of inner row from square edge

    img = np.ones((sq_px, sq_px, 3), dtype=np.uint8) * 255  # white background

    # Draw square border
    cv2.rectangle(img, (0, 0), (sq_px - 1, sq_px - 1), (0, 0, 0), 3)

    placed = []  # list of (id, centre_px_x, centre_px_y, tag_px)

    def place_tag(tag_id: int, cx: int, cy: int):
        """Render one tag centred at (cx, cy)."""
        tag_img = np.zeros((tag_px, tag_px), dtype=np.uint8)
        aruco.generateImageMarker(ARUCO_DICT, tag_id, tag_px, tag_img, 1)
        x0 = cx - tag_px // 2
        y0 = cy - tag_px // 2
        x1 = x0 + tag_px
        y1 = y0 + tag_px
        # Clip to image bounds
        ix0 = max(x0, 0); iy0 = max(y0, 0)
        ix1 = min(x1, sq_px); iy1 = min(y1, sq_px)
        sx0 = ix0 - x0; sy0 = iy0 - y0
        sx1 = sx0 + (ix1 - ix0); sy1 = sy0 + (iy1 - iy0)
        img[iy0:iy1, ix0:ix1] = tag_img[sy0:sy1, sx0:sx1, np.newaxis].repeat(3, axis=2)
        placed.append((tag_id, cx, cy))

    def tags_along_side(inset: int, n_tags: int, side: str, id_start: int):
        """Place n_tags evenly spaced along one side at given inset from edge."""
        half = tag_px // 2
        margin = inset + half  # centre of first/last tag from edge

        if side == "top":
            positions = [(int(margin + i * (sq_px - 2 * margin) / (n_tags - 1)), inset + half)
                         for i in range(n_tags)]
        elif side == "bottom":
            positions = [(int(margin + i * (sq_px - 2 * margin) / (n_tags - 1)), sq_px - inset - half)
                         for i in range(n_tags)]
        elif side == "left":
            # skip corners (already placed by top/bottom)
            positions = [(inset + half, int(margin + i * (sq_px - 2 * margin) / (n_tags - 1)))
                         for i in range(1, n_tags - 1)]
        elif side == "right":
            positions = [(sq_px - inset - half, int(margin + i * (sq_px - 2 * margin) / (n_tags - 1)))
                         for i in range(1, n_tags - 1)]
        else:
            positions = []

        for j, (cx, cy) in enumerate(positions):
            place_tag(id_start + j, cx, cy)
        return id_start + len(positions)

    # Outer row: tags fit along the full square side
    n_outer = max(3, int((square_mm - tag_mm) / (tag_mm + gap_mm)) + 1)

    # Inner row: available length is reduced by 2×inset, so fewer tags fit
    inner_inset_mm = px_to_mm(row2_inset_px)
    inner_side_mm  = square_mm - 2 * inner_inset_mm
    n_inner = max(3, int((inner_side_mm - tag_mm) / (tag_mm + gap_mm)) + 1)

    tag_id = 0

    # Outer row
    for side in ("top", "bottom", "left", "right"):
        tag_id = tags_along_side(0, n_outer, side, tag_id)

    # Inner row — fewer tags so each has proper quiet zone clearance
    for side in ("top", "bottom", "left", "right"):
        tag_id = tags_along_side(row2_inset_px, n_inner, side, tag_id)

    print(f"Total tags placed: {tag_id}  (IDs 0–{tag_id - 1})")
    print(f"Square: {square_mm:.0f}×{square_mm:.0f} mm  |  Tag: {tag_mm:.0f} mm  |  Gap: {gap_mm:.1f} mm")
    print(f"Tags per side: {n_outer} outer + {n_inner} inner")

    # ── Ruler grid inside the square ────────────────────────────────────
    # Interior starts at the inner edge of the inner tag row
    interior_inset_px = 2 * tag_px + gap_px
    interior_size_px  = sq_px - 2 * interior_inset_px

    # Draw thick black border around the interior region
    cv2.rectangle(img,
                  (interior_inset_px, interior_inset_px),
                  (interior_inset_px + interior_size_px, interior_inset_px + interior_size_px),
                  (0, 0, 0), 3)

    tick_minor_mm = 5    # thin grey lines every 5 mm
    tick_major_mm = 10   # major tick every 10 mm (labelled)

    tick_minor_px = mm_to_px(tick_minor_mm, dpi)

    minor_len = mm_to_px(1.5, dpi)
    major_len = mm_to_px(3.0, dpi)

    color_minor = (200, 200, 200)
    color_major = (150, 150, 150)
    color_tick  = (120, 120, 120)
    color_label = (60,  60,  60)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55        # larger labels
    font_thick = 1

    origin_x = interior_inset_px
    origin_y = interior_inset_px
    iend_x   = origin_x + interior_size_px
    iend_y   = origin_y + interior_size_px

    # Label margin inside the border — how far in from edge to place text
    label_margin = mm_to_px(3.5, dpi)

    def blit(canvas, patch, x, y):
        x0, y0 = max(x, 0), max(y, 0)
        x1 = min(x + patch.shape[1], canvas.shape[1])
        y1 = min(y + patch.shape[0], canvas.shape[0])
        px0, py0 = x0 - x, y0 - y
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = patch[py0:py0+(y1-y0), px0:px0+(x1-x0)]

    def rotated_label(text):
        """Render text upright then rotate 90° CCW so it reads bottom-to-top."""
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
        patch = np.ones((th + 6, tw + 6, 3), dtype=np.uint8) * 255
        cv2.putText(patch, text, (3, th + 2), font, font_scale,
                    color_label, font_thick, cv2.LINE_AA)
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    n_ticks = int(interior_size_px / tick_minor_px) + 1

    for i in range(n_ticks):
        offset_px = i * tick_minor_px
        if offset_px > interior_size_px:
            break
        is_major = (i * tick_minor_mm) % tick_major_mm == 0
        tick_len = major_len if is_major else minor_len
        dist_mm  = i * tick_minor_mm

        col_px = origin_x + offset_px   # vertical grid line X
        row_px = origin_y + offset_px   # horizontal grid line Y

        # Grid lines
        if offset_px > 0:
            line_color = color_major if is_major else color_minor
            cv2.line(img, (col_px, origin_y + 1), (col_px, iend_y - 1), line_color, 1)
            cv2.line(img, (origin_x + 1, row_px), (iend_x - 1, row_px), line_color, 1)

        # Tick marks on all four edges
        cv2.line(img, (col_px, origin_y), (col_px, origin_y + tick_len), color_tick, 1)
        cv2.line(img, (col_px, iend_y),   (col_px, iend_y - tick_len),   color_tick, 1)
        cv2.line(img, (origin_x, row_px), (origin_x + tick_len, row_px), color_tick, 1)
        cv2.line(img, (iend_x,   row_px), (iend_x - tick_len,   row_px), color_tick, 1)

        # Labels inside the border at major ticks (skip 0)
        if is_major and dist_mm > 0:
            label = f"{dist_mm}"
            rot   = rotated_label(label)
            rh, rw = rot.shape[:2]
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

            # X-axis label: upright, just inside top border, centred on grid line
            cv2.putText(img, label,
                        (col_px - tw // 2, origin_y + label_margin + th),
                        font, font_scale, color_label, font_thick, cv2.LINE_AA)

            # Y-axis label: rotated, just inside left border, centred on grid line
            blit(img, rot, origin_x + label_margin, row_px - rh // 2)

    # ── X/Y axis arrows at TL interior corner ────────────────────────────────
    arrow_len = mm_to_px(12, dpi)
    arrow_col = (40, 40, 40)
    arrow_th  = max(2, mm_to_px(0.5, dpi))
    ax_orig   = (origin_x + mm_to_px(6, dpi), origin_y + mm_to_px(6, dpi))

    # +X arrow (rightward)
    cv2.arrowedLine(img,
                    ax_orig,
                    (ax_orig[0] + arrow_len, ax_orig[1]),
                    arrow_col, arrow_th, tipLength=0.3)
    (tw, th), _ = cv2.getTextSize("X", font, 0.55, font_thick)
    cv2.putText(img, "X",
                (ax_orig[0] + arrow_len + mm_to_px(1, dpi), ax_orig[1] + th // 2),
                font, 0.55, arrow_col, font_thick, cv2.LINE_AA)

    # +Y arrow (downward — Y increases downward in board coords)
    cv2.arrowedLine(img,
                    ax_orig,
                    (ax_orig[0], ax_orig[1] + arrow_len),
                    arrow_col, arrow_th, tipLength=0.3)
    cv2.putText(img, "Y",
                (ax_orig[0] - th - mm_to_px(1, dpi), ax_orig[1] + arrow_len + th),
                font, 0.55, arrow_col, font_thick, cv2.LINE_AA)

    return img, placed


def save_pdf(img_rgb: np.ndarray, out_path: Path, square_mm: float, dpi: int = DPI):
    """Embed the board image centred on an A4 page and save as PDF."""
    # Convert to PIL
    pil_img = Image.fromarray(img_rgb)

    # A4 dimensions in mm
    a4_w_mm, a4_h_mm = 210.0, 297.0

    # Shift board up from centre to leave room for the robot base line below
    x_off_mm  = (a4_w_mm - square_mm) / 2
    y_off_mm  = (a4_h_mm - square_mm) / 2 + 40.0   # 40 mm above centre

    # Robot base line sits 15 mm below the bottom edge of the board
    robot_line_y_mm = y_off_mm - 70.0

    c = rl_canvas.Canvas(str(out_path), pagesize=A4)

    # Save PIL image to a bytes buffer and draw on PDF
    # Save to a temp PNG file (reportlab doesn't accept BytesIO directly)
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    pil_img.save(tmp.name, format="PNG", dpi=(dpi, dpi))

    c.drawImage(
        tmp.name,
        x_off_mm * mm,
        y_off_mm * mm,
        width=square_mm * mm,
        height=square_mm * mm,
    )
    os.unlink(tmp.name)

    # Label
    c.setFont("Helvetica", 8)
    c.drawCentredString(
        a4_w_mm / 2 * mm, (y_off_mm - 6) * mm,
        f"ArUco board  {square_mm:.0f}×{square_mm:.0f} mm  |  DICT_6X6_250  |  PRINT AT 100% — NO SCALING"
    )

    # 100mm scale bar so you can verify print accuracy with a ruler
    bar_x = x_off_mm * mm
    bar_y = (y_off_mm - 14) * mm
    bar_len = 100 * mm
    c.setLineWidth(1.5)
    c.line(bar_x, bar_y, bar_x + bar_len, bar_y)
    c.line(bar_x, bar_y - 2 * mm, bar_x, bar_y + 2 * mm)
    c.line(bar_x + bar_len, bar_y - 2 * mm, bar_x + bar_len, bar_y + 2 * mm)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(bar_x + bar_len / 2, bar_y + 3 * mm, "◄─────────── 100 mm ───────────►")

    # ── Robot base placement line ─────────────────────────────────────────────
    # A dashed horizontal line spanning the full board width with a label.
    # Place the robot arm base tip so it sits on this line, centred left-right.
    line_x0 = x_off_mm * mm
    line_x1 = (x_off_mm + square_mm) * mm
    line_y  = robot_line_y_mm * mm
    c.setDash([4, 3])          # 4pt on, 3pt off
    c.setLineWidth(1.2)
    c.setStrokeColorRGB(0.8, 0.1, 0.1)
    c.line(line_x0, line_y, line_x1, line_y)
    c.setDash()                # reset dash

    # Small tick at centre
    centre_x = a4_w_mm / 2 * mm
    c.setLineWidth(1.0)
    c.line(centre_x, line_y - 3 * mm, centre_x, line_y + 3 * mm)

    c.setFillColorRGB(0.8, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(centre_x, line_y - 5.5 * mm,
                        "▲  ROBOT BASE TIP — place here, centred  ▲")
    c.setFillColorRGB(0, 0, 0)   # reset fill

    c.save()
    print(f"PDF saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-mm", type=float, default=180.0,
                        help="Side length of the square in mm (default: 180)")
    parser.add_argument("--tag-mm",    type=float, default=16.0,
                        help="Side length of each ArUco tag in mm (default: 16)")
    parser.add_argument("--gap-mm",    type=float, default=2.0,
                        help="Gap between tag rows and between tags (default: 2)")
    parser.add_argument("--out",       type=str,   default="aruco_board.pdf")
    parser.add_argument("--dpi",       type=int,   default=300)
    args = parser.parse_args()

    img, placed = draw_board(
        square_mm=args.square_mm,
        tag_mm=args.tag_mm,
        gap_mm=args.gap_mm,
        dpi=args.dpi,
    )

    out_path = Path(args.out)
    save_pdf(img, out_path, args.square_mm, args.dpi)

    # Also save PNG preview
    png_path = out_path.with_suffix(".png")
    cv2.imwrite(str(png_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"PNG preview saved to {png_path}")


if __name__ == "__main__":
    main()
