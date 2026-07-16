"""make_bordered_tags.py — Generate a PDF sheet of bordered ArUco tags.

Layout
------
• Page 1 — General use tags (IDs 100–119), 35 mm total size, 5 per row
• Page 1 (continued) — Cube-replacement tags:
    ID 120 @ 16 mm  — for 20 mm calibration cube
    ID 130 @ 25 mm  — for 30 mm calibration cube
    ID 150 @ 40 mm  — for 50 mm calibration cube
• Scale bar at the bottom of every page.

Print at 100 % / no scaling so physical sizes are correct.

Usage:
    python make_bordered_tags.py
    python make_bordered_tags.py --dpi 300 --out bordered_tags.pdf
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

# ── Constants ─────────────────────────────────────────────────────────────────
DPI        = 300
A4_W_MM    = 210.0
A4_H_MM    = 297.0
MARGIN_MM  = 10.0
TAG_GAP_MM = 3.0

# General-use bordered tags
GENERAL_IDS      = list(range(100, 120))   # 20 tags
GENERAL_TOTAL_MM = 35.0                    # physical size (total incl. border)

# Cube-replacement bordered tags: (tag_id, total_size_mm, cube_size_mm)
CUBE_TAGS = [
    (120, 16.0, 20),
    (130, 25.0, 30),
    (150, 40.0, 50),
]

# This is the ratio  total_px / aruco_px  with the default param set at 200px:
#   margin_per_side = border(10) + gap(8) + inner(6) + quiet(24) = 48
#   total = 200 + 2*48 = 296  →  ratio = 296/200 = 1.48
_RATIO = 296 / 200


def mm_to_px(v, dpi=DPI):
    return int(round(v / 25.4 * dpi))


def make_tag_image(tag_id: int, total_mm: float, dpi: int = DPI) -> np.ndarray:
    """Return an RGB image of a bordered ArUco tag at the given physical total size."""
    total_px = mm_to_px(total_mm, dpi)
    aruco_px = max(32, int(total_px / _RATIO))   # size of ArUco bit-grid
    s        = aruco_px / 200                     # scale factor vs default params
    img_gray = generate_bordered_tag(
        tag_id          = tag_id,
        tag_size_px     = aruco_px,
        quiet_zone_px   = max(4,  int(24 * s)),
        border_px       = max(2,  int(10 * s)),
        double_gap_px   = max(1,  int( 8 * s)),
        double_border_px= max(1,  int( 6 * s)),
    )
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    # Snap to exact total_px so grid placement is pixel-perfect
    if img_rgb.shape[0] != total_px:
        img_rgb = cv2.resize(img_rgb, (total_px, total_px),
                             interpolation=cv2.INTER_NEAREST)
    return img_rgb


def draw_scale_bar(page_img, dpi=DPI):
    """Draw a 50 mm scale bar near the bottom of page_img (in place)."""
    a4h_px = mm_to_px(A4_H_MM, dpi)
    bx0    = mm_to_px(MARGIN_MM, dpi)
    bx1    = bx0 + mm_to_px(50, dpi)
    by     = a4h_px - mm_to_px(5, dpi)
    tick   = mm_to_px(2, dpi)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    gray   = (80, 80, 80)
    cv2.line(page_img, (bx0, by), (bx1, by), gray, 2)
    cv2.line(page_img, (bx0, by - tick), (bx0, by + tick), gray, 2)
    cv2.line(page_img, (bx1, by - tick), (bx1, by + tick), gray, 2)
    cv2.putText(page_img,
                "50 mm  |  PRINT AT 100%  NO SCALING",
                (bx0 + mm_to_px(3, dpi), by - mm_to_px(2, dpi)),
                font, 0.30, gray, 1, cv2.LINE_AA)


def new_page(dpi=DPI):
    a4w = mm_to_px(A4_W_MM, dpi)
    a4h = mm_to_px(A4_H_MM, dpi)
    img = np.ones((a4h, a4w, 3), dtype=np.uint8) * 245
    return img


def build_pages(dpi=DPI):
    font = cv2.FONT_HERSHEY_SIMPLEX
    page = new_page(dpi)
    pages = [page]

    # Title
    margin_px = mm_to_px(MARGIN_MM, dpi)
    cv2.putText(page,
                "Bordered ArUco Tags (DICT_6X6_250)  —  top-left L-indicator shows orientation",
                (margin_px, mm_to_px(7, dpi)),
                font, 0.38, (30, 30, 30), 1, cv2.LINE_AA)

    cur_y_mm = MARGIN_MM + 9   # below title

    # ── Section 1: General-use tags (35 mm, IDs 100–119) ─────────────────────
    section_label = f"General-use markers — {GENERAL_TOTAL_MM:.0f} mm total  (IDs 100–119)"
    cv2.putText(page, section_label,
                (margin_px, mm_to_px(cur_y_mm, dpi)),
                font, 0.32, (60, 60, 200), 1, cv2.LINE_AA)
    cur_y_mm += 4

    tags_per_row  = 5
    tag_step_mm   = GENERAL_TOTAL_MM + TAG_GAP_MM
    tag_px        = mm_to_px(GENERAL_TOTAL_MM, dpi)

    for i, tag_id in enumerate(GENERAL_IDS):
        col = i % tags_per_row
        row = i // tags_per_row
        x_mm = MARGIN_MM + col * tag_step_mm
        y_mm = cur_y_mm  + row * tag_step_mm
        x_px = mm_to_px(x_mm, dpi)
        y_px = mm_to_px(y_mm, dpi)

        img  = make_tag_image(tag_id, GENERAL_TOTAL_MM, dpi)
        page[y_px : y_px + tag_px, x_px : x_px + tag_px] = img

        # ID label below tag
        cv2.putText(page, f"ID {tag_id}",
                    (x_px, y_px + tag_px + mm_to_px(3.5, dpi)),
                    font, 0.26, (60, 60, 60), 1, cv2.LINE_AA)

    n_rows       = (len(GENERAL_IDS) + tags_per_row - 1) // tags_per_row
    cur_y_mm    += n_rows * tag_step_mm + 8   # gap after section

    # ── Section 2: Cube-replacement tags ─────────────────────────────────────
    cv2.putText(page, "Cube-replacement markers  —  sized to fit on calibration cubes",
                (margin_px, mm_to_px(cur_y_mm, dpi)),
                font, 0.32, (60, 60, 200), 1, cv2.LINE_AA)
    cur_y_mm += 4

    max_cube_mm  = max(t for _, t, _ in CUBE_TAGS)
    x_cursor_mm  = MARGIN_MM

    for tag_id, total_mm, cube_mm in CUBE_TAGS:
        t_px = mm_to_px(total_mm, dpi)
        x_px = mm_to_px(x_cursor_mm, dpi)
        y_px = mm_to_px(cur_y_mm + (max_cube_mm - total_mm) / 2, dpi)  # bottom-align

        img = make_tag_image(tag_id, total_mm, dpi)
        page[y_px : y_px + t_px, x_px : x_px + t_px] = img

        # Labels
        lbl1 = f"ID {tag_id}"
        lbl2 = f"{total_mm:.0f} mm total"
        lbl3 = f"for {cube_mm} mm cube"
        base_y = mm_to_px(cur_y_mm + max_cube_mm, dpi) + mm_to_px(3, dpi)
        cv2.putText(page, lbl1, (x_px, base_y),                    font, 0.28, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(page, lbl2, (x_px, base_y + mm_to_px(4, dpi)), font, 0.26, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(page, lbl3, (x_px, base_y + mm_to_px(8, dpi)), font, 0.26, (80, 80, 80), 1, cv2.LINE_AA)

        x_cursor_mm += total_mm + 10

    draw_scale_bar(page, dpi)
    return pages


def build_uniform_pages(tag_mm: float, count: int, id_start: int, dpi=DPI):
    """Grid pages of identical-size bordered tags (IDs id_start..id_start+count-1)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_px = mm_to_px(MARGIN_MM, dpi)

    step_mm      = tag_mm + TAG_GAP_MM + 4      # extra 4mm for the ID label
    tags_per_row = int((A4_W_MM - 2 * MARGIN_MM) // (tag_mm + TAG_GAP_MM))
    top_mm       = MARGIN_MM + 9                # below title
    rows_per_pg  = int((A4_H_MM - top_mm - 12) // step_mm)
    per_page     = tags_per_row * rows_per_pg

    tag_px = mm_to_px(tag_mm, dpi)
    pages  = []
    for p in range((count + per_page - 1) // per_page):
        page = new_page(dpi)
        cv2.putText(page,
                    f"Bordered ArUco tags (DICT_6X6_250) — {tag_mm:.0f} mm total — PRINT AT 100%",
                    (margin_px, mm_to_px(7, dpi)),
                    font, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
        for k in range(per_page):
            i = p * per_page + k
            if i >= count:
                break
            tag_id = id_start + i
            col = k % tags_per_row
            row = k // tags_per_row
            x_px = mm_to_px(MARGIN_MM + col * (tag_mm + TAG_GAP_MM), dpi)
            y_px = mm_to_px(top_mm + row * step_mm, dpi)
            page[y_px : y_px + tag_px, x_px : x_px + tag_px] = \
                make_tag_image(tag_id, tag_mm, dpi)
            cv2.putText(page, f"{tag_id}",
                        (x_px, y_px + tag_px + mm_to_px(3, dpi)),
                        font, 0.26, (60, 60, 60), 1, cv2.LINE_AA)
        draw_scale_bar(page, dpi)
        pages.append(page)
    return pages


def save_pdf(pages, out_path, dpi=DPI):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    for page in pages:
        pil = Image.fromarray(page)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        pil.save(tmp.name, format="PNG", dpi=(dpi, dpi))
        c.drawImage(tmp.name, 0, 0, width=210 * mm, height=297 * mm)
        c.showPage()
        os.unlink(tmp.name)
    c.save()
    print(f"PDF saved: {out_path}")

    # Also save PNG previews
    for i, page in enumerate(pages):
        png = out_path.with_name(out_path.stem + f"_page{i+1}.png")
        cv2.imwrite(str(png), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
        print(f"PNG saved: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=str, default="printables/bordered_tags.pdf")
    ap.add_argument("--tag-mm",   type=float, default=None,
                    help="Uniform-sheet mode: fill pages with tags of this total size (mm)")
    ap.add_argument("--count",    type=int,   default=None,
                    help="Uniform-sheet mode: how many tags (default: fill one page)")
    ap.add_argument("--id-start", type=int,   default=100,
                    help="Uniform-sheet mode: first tag ID (default 100)")
    args = ap.parse_args()

    if args.tag_mm is not None:
        tags_per_row = int((A4_W_MM - 2 * MARGIN_MM) // (args.tag_mm + TAG_GAP_MM))
        rows = int((A4_H_MM - MARGIN_MM - 21) // (args.tag_mm + TAG_GAP_MM + 4))
        count = args.count if args.count is not None else tags_per_row * rows
        if args.id_start + count > 250:
            count = 250 - args.id_start
            print(f"Clamped to {count} tags (DICT_6X6_250 has IDs 0–249)")
        pages = build_uniform_pages(args.tag_mm, count, args.id_start, dpi=args.dpi)
        save_pdf(pages, args.out, dpi=args.dpi)
        print(f"\n{count} bordered tags @ {args.tag_mm:.0f} mm, "
              f"IDs {args.id_start}–{args.id_start + count - 1}")
        return

    pages = build_pages(dpi=args.dpi)
    save_pdf(pages, args.out, dpi=args.dpi)

    print("\nTag IDs in this PDF:")
    print(f"  General use  (35 mm): IDs {GENERAL_IDS[0]}–{GENERAL_IDS[-1]}")
    for tid, tmm, cmm in CUBE_TAGS:
        print(f"  ID {tid:3d}  ({tmm:.0f} mm total)  — for {cmm} mm cube")


if __name__ == "__main__":
    main()
