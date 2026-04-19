"""board.py — ArUco calibration board generation with robot tip markers.

Generates a PDF of the calibration board used for tag-based robot localisation.
The board contains ArUco border tags on all sides, and optionally markers
indicating where the robot arm base tips should be placed.

Example
-------
>>> from lerobot_tagged import create_aruco_board_pdf, BoardConfig
>>> cfg = BoardConfig()
>>> create_aruco_board_pdf("board.pdf", cfg)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas


ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Physical dimensions in mm
A4_W_MM = 210.0
A4_H_MM = 297.0


@dataclass
class BoardConfig:
    """Physical configuration of the calibration board."""

    square_mm: float = 180.0
    """Total board side length in mm."""

    tag_mm: float = 16.0
    """Side length of each border ArUco tag in mm."""

    gap_mm: float = 2.0
    """Gap between inner and outer border tag rows in mm."""

    n_outer: int = 10
    """Number of tags along each outer border edge."""

    n_inner: int = 8
    """Number of tags along each inner border edge."""

    dpi: int = 300
    """Print resolution."""

    # Robot placement markers
    show_robot_tips: bool = True
    """Draw markers showing where the robot arm base tips should be placed."""

    robot_tip_left_mm: Tuple[float, float] = field(default_factory=lambda: (80.0, 180.0))
    """(x, y) position of the robot base left tip in board mm. Origin = board TL."""

    robot_tip_right_mm: Tuple[float, float] = field(default_factory=lambda: (100.0, 180.0))
    """(x, y) position of the robot base right tip in board mm. Origin = board TL."""

    robot_tip_radius_mm: float = 2.0
    """Radius of the tip marker circles in mm."""


def border_tag_tl_positions(cfg: BoardConfig | None = None) -> dict:
    """Return the top-left corner position (in board mm) of every border tag.

    The origin is the board top-left corner. Coordinates increase right and down.
    Used by :meth:`~lerobot_tagged.Detector.calibrate` to build the homography.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping of tag ID → ``(x_mm, y_mm)`` of its top-left corner.
    """
    if cfg is None:
        cfg = BoardConfig()
    positions: dict = {}
    tag_id = 0
    half = cfg.tag_mm / 2
    for inset, n in [(0, cfg.n_outer), (cfg.tag_mm + cfg.gap_mm, cfg.n_inner)]:
        margin = inset + half
        span = cfg.square_mm - 2 * margin
        for side in ("top", "bottom", "left", "right"):
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                along = margin + t * span
                if side == "top":
                    cx, cy = along, margin
                elif side == "bottom":
                    cx, cy = along, cfg.square_mm - margin
                elif side == "left":
                    if i == 0 or i == n - 1:
                        continue
                    cx, cy = margin, along
                elif side == "right":
                    if i == 0 or i == n - 1:
                        continue
                    cx, cy = cfg.square_mm - margin, along
                else:
                    continue
                positions[tag_id] = (cx - half, cy - half)
                tag_id += 1
    return positions


def _mm_to_px(value_mm: float, dpi: int) -> int:
    return int(round(value_mm / 25.4 * dpi))


def _render_board_image(cfg: BoardConfig) -> np.ndarray:
    """Render the board as a numpy BGR image at cfg.dpi resolution."""
    px = lambda v: _mm_to_px(v, cfg.dpi)

    board_px = px(cfg.square_mm)
    img = np.full((board_px, board_px, 3), 255, dtype=np.uint8)

    tag_half = px(cfg.tag_mm) // 2

    def side_centres(inset_mm, n, side):
        half_mm = cfg.tag_mm / 2
        margin = inset_mm + half_mm
        span = cfg.square_mm - 2 * margin
        out = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            along = margin + t * span
            if side == "top":
                out.append((along, inset_mm + half_mm))
            elif side == "bottom":
                out.append((along, cfg.square_mm - inset_mm - half_mm))
            elif side == "left":
                if 0 < i < n - 1:
                    out.append((inset_mm + half_mm, along))
            elif side == "right":
                if 0 < i < n - 1:
                    out.append((cfg.square_mm - inset_mm - half_mm, along))
        return out

    tag_id = 0
    for inset, n in [(0, cfg.n_outer), (cfg.tag_mm + cfg.gap_mm, cfg.n_inner)]:
        for side in ("top", "bottom", "left", "right"):
            for cx_mm, cy_mm in side_centres(inset, n, side):
                tag_img = aruco.generateImageMarker(ARUCO_DICT, tag_id, px(cfg.tag_mm))
                tag_bgr = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
                cx_px = px(cx_mm)
                cy_px = px(cy_mm)
                x0 = cx_px - tag_half
                y0 = cy_px - tag_half
                x1 = x0 + px(cfg.tag_mm)
                y1 = y0 + px(cfg.tag_mm)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(board_px, x1), min(board_px, y1)
                th = px(cfg.tag_mm)
                img[y0:y1, x0:x1] = tag_bgr[:y1 - y0, :x1 - x0]
                tag_id += 1

    # Board border
    cv2.rectangle(img, (0, 0), (board_px - 1, board_px - 1), (0, 0, 0), 2)

    # Interior rectangle
    inset_px = px(2 * cfg.tag_mm + cfg.gap_mm)
    cv2.rectangle(img,
                  (inset_px, inset_px),
                  (board_px - inset_px, board_px - inset_px),
                  (180, 180, 180), 1)

    # Robot tip markers
    if cfg.show_robot_tips:
        r_px = px(cfg.robot_tip_radius_mm)
        for tip_mm, label in [
            (cfg.robot_tip_left_mm,  "L"),
            (cfg.robot_tip_right_mm, "R"),
        ]:
            tx = px(tip_mm[0])
            ty = px(tip_mm[1])
            # Crosshair
            arm = r_px * 3
            cv2.line(img, (tx - arm, ty), (tx + arm, ty), (200, 0, 0), max(1, r_px // 2))
            cv2.line(img, (tx, ty - arm), (tx, ty + arm), (200, 0, 0), max(1, r_px // 2))
            # Circle
            cv2.circle(img, (tx, ty), r_px, (200, 0, 0), -1)
            cv2.putText(img, label, (tx + r_px + 2, ty - r_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1, cv2.LINE_AA)

    return img


def create_aruco_board_pdf(
    output_path: str | Path,
    cfg: BoardConfig | None = None,
) -> Path:
    """Generate a PDF of the ArUco calibration board.

    The board is centred on an A4 page. If ``cfg.show_robot_tips`` is True,
    two markers are drawn at the bottom of the board indicating where the
    robot arm base tips should be placed.

    Parameters
    ----------
    output_path:
        Path for the output PDF file.
    cfg:
        Board configuration. Uses defaults if not provided.

    Returns
    -------
    Path
        Resolved path to the written PDF.
    """
    if cfg is None:
        cfg = BoardConfig()

    output_path = Path(output_path)
    board_img = _render_board_image(cfg)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, board_img)

    try:
        c = rl_canvas.Canvas(str(output_path), pagesize=(A4_W_MM * mm, A4_H_MM * mm))

        board_pdf_mm = cfg.square_mm
        x_off = (A4_W_MM - board_pdf_mm) / 2 * mm
        y_off = (A4_H_MM - board_pdf_mm) / 2 * mm

        c.drawImage(tmp_path, x_off, y_off,
                    width=board_pdf_mm * mm, height=board_pdf_mm * mm)

        # Scale bar
        bar_x = x_off
        bar_y = y_off - 8 * mm
        bar_len = 10 * mm
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(0.5)
        c.line(bar_x, bar_y, bar_x + bar_len, bar_y)
        c.line(bar_x, bar_y - 1 * mm, bar_x, bar_y + 1 * mm)
        c.line(bar_x + bar_len, bar_y - 1 * mm, bar_x + bar_len, bar_y + 1 * mm)
        c.setFont("Helvetica", 6)
        c.drawCentredString(bar_x + bar_len / 2, bar_y - 4 * mm, "10 mm")

        c.save()
    finally:
        os.unlink(tmp_path)

    return output_path.resolve()
