"""Generate a printable A4 PDF of individual ArUco tags for cut-out testing.

Tags are laid out in a grid with their ID printed below each one.
IDs start at 100 (above the board's 0-71 range) to avoid conflicts.

Usage:
    python make_aruco_tags.py
    python make_aruco_tags.py --tag-mm 20 --cols 6 --id-start 100
"""
import argparse
import os
import tempfile
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
DPI = 300


def mm_to_px(mm_val: float, dpi: int = DPI) -> int:
    return int(round(mm_val / 25.4 * dpi))


def make_tag_sheet(
    tag_mm: float = 20.0,
    gap_mm: float = 5.0,
    cols: int = 6,
    id_start: int = 100,
    dpi: int = DPI,
) -> tuple[np.ndarray, int]:
    a4_w_mm, a4_h_mm = 210.0, 297.0
    margin_mm = 10.0

    tag_px = mm_to_px(tag_mm, dpi)
    gap_px = mm_to_px(gap_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)
    label_h_px = mm_to_px(5.0, dpi)  # space for ID label below tag

    cell_px = tag_px + gap_px

    sheet_w_px = mm_to_px(a4_w_mm, dpi)
    sheet_h_px = mm_to_px(a4_h_mm, dpi)

    usable_w = sheet_w_px - 2 * margin_px
    usable_h = sheet_h_px - 2 * margin_px

    rows = min(2, usable_h // (cell_px + label_h_px))  # max 2 rows (~8 tags)

    img = np.ones((sheet_h_px, sheet_w_px, 3), dtype=np.uint8) * 255

    tag_id = id_start
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thick = 1

    for row in range(rows):
        for col in range(cols):
            x0 = margin_px + col * cell_px
            y0 = margin_px + row * (cell_px + label_h_px)

            if x0 + tag_px > sheet_w_px - margin_px:
                continue
            if y0 + tag_px + label_h_px > sheet_h_px - margin_px:
                continue

            # Draw tag
            tag_img = np.zeros((tag_px, tag_px), dtype=np.uint8)
            aruco.generateImageMarker(ARUCO_DICT, tag_id, tag_px, tag_img, 1)
            img[y0:y0+tag_px, x0:x0+tag_px] = tag_img[:, :, np.newaxis].repeat(3, axis=2)

            # Draw thin border around tag
            cv2.rectangle(img, (x0, y0), (x0+tag_px, y0+tag_px), (180, 180, 180), 1)

            # Cut lines (dashed feel via small ticks at corners)
            tick = mm_to_px(1.5, dpi)
            for dx, dy in [(0,0),(tag_px,0),(0,tag_px),(tag_px,tag_px)]:
                cv2.line(img, (x0+dx-tick if dx else x0, y0+dy), (x0+dx+tick if not dx else x0+dx, y0+dy), (150,150,150), 1)
                cv2.line(img, (x0+dx, y0+dy-tick if dy else y0), (x0+dx, y0+dy+tick if not dy else y0+dy), (150,150,150), 1)

            # ID label
            label = f"ID {tag_id}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            cv2.putText(img, label,
                        (x0 + (tag_px - tw) // 2, y0 + tag_px + th + mm_to_px(1, dpi)),
                        font, font_scale, (80, 80, 80), font_thick, cv2.LINE_AA)

            tag_id += 1

    n_tags = tag_id - id_start
    print(f"Tags: {n_tags}  (IDs {id_start}–{tag_id-1})  |  {cols} cols × {rows} rows  |  {tag_mm:.0f}mm each")
    return img, n_tags


def save_pdf(img: np.ndarray, out_path: Path, dpi: int = DPI):
    pil_img = Image.fromarray(img)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    pil_img.save(tmp.name, format="PNG", dpi=(dpi, dpi))

    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    c.drawImage(tmp.name, 0, 0, width=210*mm, height=297*mm)
    c.setFont("Helvetica", 7)
    c.drawCentredString(105*mm, 5*mm, f"ArUco tags  DICT_6X6_250  —  cut out and use for positioning tests")
    c.save()
    os.unlink(tmp.name)
    print(f"PDF saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-mm",   type=float, default=12.0)
    parser.add_argument("--gap-mm",   type=float, default=5.0)
    parser.add_argument("--cols",     type=int,   default=4)
    parser.add_argument("--id-start", type=int,   default=100)
    parser.add_argument("--out",      type=str,   default="aruco_tags.pdf")
    parser.add_argument("--dpi",      type=int,   default=300)
    args = parser.parse_args()

    img, _ = make_tag_sheet(
        tag_mm=args.tag_mm,
        gap_mm=args.gap_mm,
        cols=args.cols,
        id_start=args.id_start,
        dpi=args.dpi,
    )

    out_path = Path(args.out)
    save_pdf(img, out_path, args.dpi)

    png_path = out_path.with_suffix(".png")
    cv2.imwrite(str(png_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"PNG preview saved to {png_path}")


if __name__ == "__main__":
    main()
