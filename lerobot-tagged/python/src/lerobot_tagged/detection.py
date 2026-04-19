"""detection.py — ArUco tag detection.

Example
-------
>>> import cv2
>>> from lerobot_tagged import Tag, Detector
>>>
>>> detector = Detector([
...     Tag(101, offset=(-5.0, 1.0), width_mm=25, height_mm=75, depth_mm=25),
... ])
>>>
>>> cap = cv2.VideoCapture(0)
>>> _, frame = cap.read()
>>> detector.calibrate(frame)   # point camera at board, run once per session
>>>
>>> while True:
...     _, frame = cap.read()
...     for loc in detector.find(frame):
...         print(loc.tag_id, loc.pos)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


_ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

_DETECTOR_PARAMS = aruco.DetectorParameters()
_DETECTOR_PARAMS.adaptiveThreshWinSizeMin  = 3
_DETECTOR_PARAMS.adaptiveThreshWinSizeMax  = 53
_DETECTOR_PARAMS.adaptiveThreshWinSizeStep = 4
_DETECTOR_PARAMS.errorCorrectionRate       = 0.7

_DETECTOR = aruco.ArucoDetector(_ARUCO_DICT, _DETECTOR_PARAMS)


@dataclass
class Tag:
    """A tag to detect, with optional pose correction and object dimensions.

    Parameters
    ----------
    id:
        ArUco tag ID to search for.
    offset:
        ``(delta_x, delta_y)`` in mm added to the raw position to correct for
        residual perspective error. Defaults to ``(0, 0)``.
    width_mm:
        Physical width of the object this tag is attached to (mm). Optional.
    height_mm:
        Physical height of the object (mm). Optional.
    depth_mm:
        Physical depth of the object (mm). Optional.
    stl_path:
        Path to an STL mesh for this object. Reserved for future use.
    """
    id: int
    offset: Tuple[float, float] = (0.0, 0.0)
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    depth_mm: Optional[float] = None
    stl_path: Optional[str] = None


@dataclass
class TagLocation:
    """Result of a single tag detection."""

    tag_id: int
    """The detected ArUco tag ID."""

    corners_px: np.ndarray
    """(4, 2) float32 — tag corners in image pixel space [TL, TR, BR, BL]."""

    center_px: Tuple[float, float]
    """(x, y) centre of the tag in image pixel space."""

    tl_px: Tuple[float, float]
    """(x, y) top-left corner of the tag in image pixel space."""

    interior_mm: Tuple[float, float]
    """(x, y) position in board mm (origin = board top-left corner)."""

    pos: Tuple[float, float]
    """(x, y) position after applying the tag's offset correction."""


class Detector:
    """Detects a fixed set of ArUco tags (and in future, robot arm joints) in
    camera frames.

    Call :meth:`calibrate` once with a frame that shows the full board, then
    call :meth:`find` on every subsequent frame.

    Parameters
    ----------
    tags:
        List of :class:`Tag` objects — which IDs to detect and their individual
        correction offsets and object dimensions.

    Example
    -------
    >>> detector = Detector([
    ...     Tag(101, offset=(-5.0, 1.0), width_mm=25, height_mm=75, depth_mm=25),
    ... ])
    >>> detector.calibrate(frame)
    >>> locations = detector.find(frame)
    """

    def __init__(self, tags: List[Tag]) -> None:
        self.tags: Dict[int, Tag] = {t.id: t for t in tags}
        self._H: Optional[np.ndarray] = None

    def calibrate(self, frame: np.ndarray, cfg=None) -> None:
        """Compute the homography from a frame showing the full calibration board.

        Detects all visible border tags, matches them to their known positions
        on the board, and computes a homography that maps image pixels directly
        to board mm (origin = board top-left corner).

        Parameters
        ----------
        frame:
            BGR or grayscale image with the full board visible.
        cfg:
            :class:`~lerobot_tagged.BoardConfig` describing the board layout.
            Uses defaults if not provided.

        Raises
        ------
        RuntimeError
            If fewer than 4 border tags are detected.
        """
        from .board import BoardConfig, border_tag_tl_positions

        if cfg is None:
            cfg = BoardConfig()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        det_corners, det_ids, _ = _DETECTOR.detectMarkers(gray)

        if det_ids is None:
            raise RuntimeError("No tags detected. Ensure the board is fully visible.")

        known = border_tag_tl_positions(cfg)
        src_pts, dst_pts = [], []
        for i, tid in enumerate(det_ids.flatten()):
            if int(tid) in known:
                src_pts.append(det_corners[i][0][0])   # TL corner in pixels
                dst_pts.append(known[int(tid)])          # TL corner in board mm

        if len(src_pts) < 4:
            raise RuntimeError(
                f"Only {len(src_pts)} border tag(s) matched — need at least 4. "
                "Ensure the board is fully visible and well-lit."
            )

        self._H, _ = cv2.findHomography(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            cv2.RANSAC,
        )

    def find(self, frame: np.ndarray) -> List[TagLocation]:
        """Detect all registered tags in a camera frame.

        Parameters
        ----------
        frame:
            BGR or grayscale camera image.

        Returns
        -------
        List[TagLocation]
            One entry per detected tag. Tags not visible are omitted.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called yet.
        """
        if self._H is None:
            raise RuntimeError("Call calibrate() before find().")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        det_corners, det_ids, _ = _DETECTOR.detectMarkers(gray)

        if det_ids is None:
            return []

        results: List[TagLocation] = []
        for i, tid in enumerate(det_ids.flatten()):
            tag = self.tags.get(int(tid))
            if tag is None:
                continue

            corners = det_corners[i].reshape(4, 2)
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())
            tl = (float(corners[0, 0]), float(corners[0, 1]))

            board_pt = cv2.perspectiveTransform(
                corners[0].astype(np.float32).reshape(1, 1, 2), self._H
            ).reshape(2)
            interior_mm = (float(board_pt[0]), float(board_pt[1]))
            pos = (
                interior_mm[0] + tag.offset[0],
                interior_mm[1] + tag.offset[1],
            )

            results.append(TagLocation(
                tag_id=tag.id,
                corners_px=corners,
                center_px=(cx, cy),
                tl_px=tl,
                interior_mm=interior_mm,
                pos=pos,
            ))

        return results
