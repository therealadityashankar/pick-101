"""make_jenga_tag.py — Print a sheet of bordered ArUco tags for jenga stack calibration.

The tag (ID 101) is sized to fit on the top face of a jenga block (25 × 75 mm).
Total bordered tag size: 22 mm — fits within the 25 mm width with 1.5 mm margin each side.
Multiple copies are printed so you have spares.

Usage:
    uv run python make_jenga_tag.py
    uv run python make_jenga_tag.py --out jenga_tag.pdf
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

from bordered_aruco import generate_bordered_tag

DPI        = 300
A4_W_MM    = 210.0
A4_H_MM    = 297.0
MARGIN_MM  = 15.0

TAGS         = [101, 102]   # 101 = jenga stack, 102 = flat on board (h=0)
TAG_TOTAL_MM = 22.0   # fits on 25mm-wide jenga block top with ~1.5mm margin each side

# Ratio from bordered_aruco default params: total/aruco = 296/200
_RATIO = 296 / 200


def mm_to_px(v, dpi=DPI):
    return int(round(v / 25.4 * dpi))


def make_tag_image_id(tag_id: int, total_mm: float, dpi: int = DPI) -> np.ndarray:
    total_px = mm_to_px(total_mm, dpi)
    aruco_px = max(32, int(total_px / _RATIO))
    s        = aruco_px / 200
    img_gray = generate_bordered_tag(
        tag_id           = tag_id,
        tag_size_px      = aruco_px,
        quiet_zone_px    = max(4, int(24 * s)),
        border_px        = max(2, int(10 * s)),
        double_gap_px    = max(1, int( 8 * s)),
        double_border_px = max(1, int( 6 * s)),
    )
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if img_rgb.shape[0] != total_px:
        img_rgb = cv2.resize(img_rgb, (total_px, total_px), interpolation=cv2.INTER_NEAREST)
    return img_rgb


def build_page(dpi=DPI):
    a4w = mm_to_px(A4_W_MM, dpi)
    a4h = mm_to_px(A4_H_MM, dpi)
    page = np.ones((a4h, a4w, 3), dtype=np.uint8) * 245
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_px = mm_to_px(MARGIN_MM, dpi)

    # Title
    cv2.putText(page,
                f"Jenga calibration tags — IDs {TAGS}  ({TAG_TOTAL_MM:.0f}mm total)  "
                f"fit on 25mm jenga block top",
                (margin_px, mm_to_px(10, dpi)),
                font, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(page,
                "Top-left L-indicator must face the interior of the board (away from you).",
                (margin_px, mm_to_px(15, dpi)),
                font, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    tag_px  = mm_to_px(TAG_TOTAL_MM, dpi)
    gap_px  = mm_to_px(12, dpi)
    labels  = ["for jenga stack (h>0)", "for flat placement (h=0)"]
    start_y = mm_to_px(20, dpi)

    for i, (tag_id, label) in enumerate(zip(TAGS, labels)):
        tag_img = make_tag_image_id(tag_id, TAG_TOTAL_MM, dpi)
        x = margin_px + i * (tag_px + gap_px)
        y = start_y
        page[y : y + tag_px, x : x + tag_px] = tag_img
        cv2.putText(page, f"ID {tag_id}  —  {TAG_TOTAL_MM:.0f}mm  —  {label}",
                    (x, y + tag_px + mm_to_px(4, dpi)),
                    font, 0.32, (60, 60, 60), 1, cv2.LINE_AA)

    print(f"Tags: {TAGS}, {TAG_TOTAL_MM:.0f}mm total each")

    # Scale bar
    bx0 = margin_px
    bx1 = bx0 + mm_to_px(50, dpi)
    by  = a4h - mm_to_px(6, dpi)
    tick = mm_to_px(2, dpi)
    cv2.line(page, (bx0, by), (bx1, by), (80, 80, 80), 2)
    cv2.line(page, (bx0, by - tick), (bx0, by + tick), (80, 80, 80), 2)
    cv2.line(page, (bx1, by - tick), (bx1, by + tick), (80, 80, 80), 2)
    cv2.putText(page, "50 mm  |  PRINT AT 100%  NO SCALING",
                (bx0 + mm_to_px(3, dpi), by - mm_to_px(2, dpi)),
                font, 0.30, (80, 80, 80), 1, cv2.LINE_AA)

    return page


def save_pdf(page, out_path, dpi=DPI):
    out_path = Path(out_path)
    pil = Image.fromarray(page)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    pil.save(tmp.name, format="PNG", dpi=(dpi, dpi))
    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    c.drawImage(tmp.name, 0, 0, width=210 * mm, height=297 * mm)
    c.save()
    os.unlink(tmp.name)
    print(f"PDF saved: {out_path}")

    png = out_path.with_suffix('.png')
    cv2.imwrite(str(png), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    print(f"PNG saved: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=str, default="jenga_tag.pdf")
    args = ap.parse_args()

    page = build_page(dpi=args.dpi)
    save_pdf(page, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
