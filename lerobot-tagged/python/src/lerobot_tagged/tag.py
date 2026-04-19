"""tag.py — Bordered ArUco tag generation.

A bordered ArUco tag has a thick outer border on all four sides, plus an
L-shaped double-border indicator at the top-left corner that unambiguously
identifies orientation.

Example
-------
>>> from lerobot_tagged import generate_tag, save_tag_pdf
>>>
>>> img = generate_tag(tag_id=101)          # returns (H, W) uint8 ndarray
>>> save_tag_pdf("tag_101.pdf", tag_id=101) # saves a print-ready PDF
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import cv2.aruco as aruco
import numpy as np


_ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)


@dataclass
class TagConfig:
    """Physical and rendering configuration for a bordered ArUco tag."""

    tag_size_px: int = 400
    """Pixel size of the ArUco bit grid."""

    quiet_zone_px: int = 24
    """White space between the innermost border line and the ArUco bits (px)."""

    border_px: int = 10
    """Thickness of the outer border rectangle (px)."""

    double_gap_px: int = 8
    """Gap between the outer border and the L-indicator line (px)."""

    double_border_px: int = 6
    """Thickness of the L-indicator line (px)."""

    double_length_ratio: float = 0.45
    """Fraction of the total image side length covered by each indicator arm."""

    tag_size_mm: float = 40.0
    """Physical width of the ArUco bit grid when printed (mm).
    Used to size the PDF page correctly."""


def generate_tag(
    tag_id: int,
    cfg: Optional[TagConfig] = None,
) -> np.ndarray:
    """Generate a bordered ArUco tag as a grayscale image.

    The tag has a thick outer border on all sides and an L-shaped indicator
    at the top-left corner to mark orientation unambiguously.

    Parameters
    ----------
    tag_id:
        ArUco marker ID (0–249 for DICT_4X4_250).
    cfg:
        Rendering configuration. Uses defaults if omitted.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W), dtype uint8.
    """
    if cfg is None:
        cfg = TagConfig()

    tag_img = np.zeros((cfg.tag_size_px, cfg.tag_size_px), dtype=np.uint8)
    aruco.generateImageMarker(_ARUCO_DICT, tag_id % 250, cfg.tag_size_px, tag_img, 1)

    margin = cfg.border_px + cfg.double_gap_px + cfg.double_border_px + cfg.quiet_zone_px
    total = cfg.tag_size_px + 2 * margin

    img = np.ones((total, total), dtype=np.uint8) * 255
    img[margin:margin + cfg.tag_size_px, margin:margin + cfg.tag_size_px] = tag_img

    # Outer border (all four sides)
    cv2.rectangle(img, (0, 0), (total - 1, total - 1), 0, cfg.border_px)

    # L-shaped indicator at top-left
    inner_pos = cfg.border_px + cfg.double_gap_px + cfg.double_border_px // 2
    arm_len = int(total * cfg.double_length_ratio)
    cv2.line(img, (0, inner_pos), (arm_len, inner_pos), 0, cfg.double_border_px)
    cv2.line(img, (inner_pos, 0), (inner_pos, arm_len), 0, cfg.double_border_px)

    return img


def save_tag_pdf(
    output_path: str,
    tag_id: int,
    cfg: Optional[TagConfig] = None,
) -> None:
    """Render a bordered ArUco tag and save it as a print-ready PDF.

    The tag is centred on an A4 page. Print at 100% scale — no page scaling.

    Parameters
    ----------
    output_path:
        Path to write the PDF file.
    tag_id:
        ArUco marker ID (0–249).
    cfg:
        Rendering configuration. Uses defaults if omitted.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
        import tempfile, os
    except ImportError as e:
        raise ImportError(
            "reportlab is required for PDF export. "
            'Install it with: pip install "lerobot-tagged[pdf]"'
        ) from e

    if cfg is None:
        cfg = TagConfig()

    img = generate_tag(tag_id, cfg)

    # Physical total size in mm (proportional to tag_size_mm)
    scale = cfg.tag_size_mm / cfg.tag_size_px
    total_mm = img.shape[0] * scale

    # Save image to a temp PNG, embed in PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, img)

    try:
        page_w, page_h = A4
        x = (page_w - total_mm * mm) / 2
        y = (page_h - total_mm * mm) / 2

        c = rl_canvas.Canvas(output_path, pagesize=A4)
        c.drawImage(tmp_path, x, y, width=total_mm * mm, height=total_mm * mm)
        c.save()
    finally:
        os.unlink(tmp_path)
