"""lerobot-tagged — ArUco board generation and tag-based robot arm localisation."""

from .board import create_aruco_board_pdf, BoardConfig
from .detection import Tag, Detector, TagLocation
from .recorder import Frame, Recorder
from .tag import generate_tag, save_tag_pdf, TagConfig

__all__ = [
    "create_aruco_board_pdf",
    "BoardConfig",
    "Tag",
    "Detector",
    "TagLocation",
    "Frame",
    "Recorder",
    "generate_tag",
    "save_tag_pdf",
    "TagConfig",
]
