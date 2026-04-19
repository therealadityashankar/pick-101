"""recorder.py — Record tag detections and replay in MuJoCo.

Example
-------
>>> import cv2
>>> from lerobot_tagged import Tag, Detector, Recorder
>>>
>>> detector = Detector(
...     tags=[Tag(101, offset=(-5.0, 1.0), width_mm=25, height_mm=75, depth_mm=25)],
...     H=H, scale=scale, inset_mm=16,
... )
>>>
>>> cap = cv2.VideoCapture(0)
>>> recorder = Recorder(cap, detector)
>>> recorder.record()
>>> # ... do other things while recording ...
>>> recorder.stop_recording()
>>>
>>> for frame in recorder.frames:
...     print(frame.timestamp, frame.tag_locations)
>>>
>>> recorder.output_mujoco_recording("scene.mp4")
"""

from __future__ import annotations

import importlib.resources
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection import Detector, Tag, TagLocation


# Dummy joint names for the SO-101 robot arm.
# These will be populated with real values in a future update.
_DUMMY_JOINTS: Dict[str, float] = {
    "shoulder_pan":   0.0,
    "shoulder_lift":  0.0,
    "elbow_flex":     0.0,
    "wrist_flex":     0.0,
    "wrist_roll":     0.0,
    "gripper":        0.0,
}


@dataclass
class Frame:
    """A single recorded frame.

    Attributes
    ----------
    timestamp:
        Seconds elapsed since ``record()`` was called.
    tag_locations:
        All detected tag locations in this frame.
    joint_positions:
        Robot arm joint angles in radians, keyed by joint name.
        Currently returns dummy zero values; will be populated from
        real hardware in a future update.
    end_effector:
        (x, y, z) end-effector position in world metres.
        Currently a dummy value; will be computed from joint positions.
    """
    timestamp: float
    tag_locations: List[TagLocation]
    joint_positions: Dict[str, float]
    end_effector: Tuple[float, float, float]


class Recorder:
    """Record tag detections from a camera and replay in MuJoCo.

    Parameters
    ----------
    cap:
        An open ``cv2.VideoCapture`` instance.
    detector:
        A configured :class:`~lerobot_tagged.Detector` instance.

    Example
    -------
    >>> recorder = Recorder(cap, detector)
    >>> recorder.record()        # starts background thread, non-blocking
    >>> time.sleep(5)
    >>> recorder.stop_recording()
    >>> print(len(recorder.frames), "frames recorded")
    >>> recorder.output_mujoco_recording("output.mp4")
    """

    def __init__(self, cap: cv2.VideoCapture, detector: Detector) -> None:
        self._cap = cap
        self._detector = detector
        self._frames: List[Frame] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def record(self) -> None:
        """Start recording in a background thread (non-blocking).

        Raises
        ------
        RuntimeError
            If recording is already in progress.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Recording is already in progress. Call stop_recording() first.")

        self._frames = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_recording(self) -> None:
        """Signal the recording thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def frames(self) -> List[Frame]:
        """All recorded frames, in chronological order.

        Safe to read after :meth:`stop_recording` is called.
        """
        return list(self._frames)

    def output_mujoco_recording(
        self,
        output_path: str = "mujoco_recording.mp4",
        xml_path: Optional[str] = None,
        camera: str = "topdown",
        width: int = 600,
        height: int = 600,
        fps: float = 30.0,
    ) -> None:
        """Render the recorded frames in MuJoCo and write to a video file.

        Each frame positions the tracked objects in the scene according to
        their detected ``pos`` coordinates, then renders
        from the specified camera.

        Board interior mm coordinates are mapped to MuJoCo world metres with
        X → X, Y → Y (both divided by 1000). Z is set to half the object's
        ``height_mm`` so it rests on the floor. Override with a custom
        ``xml_path`` to use your own coordinate system.

        Parameters
        ----------
        output_path:
            Path for the output video file (mp4).
        xml_path:
            Path to a MuJoCo XML scene. When omitted, an auto-generated
            scene is used that includes a box body for each tracked tag
            (sized from ``width_mm``, ``height_mm``, ``depth_mm``).
        camera:
            Name of the MuJoCo camera to render from.
        width, height:
            Output video dimensions in pixels.
        fps:
            Frames per second for the output video.
        """
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "mujoco is required for output_mujoco_recording. "
                'Install it with: pip install "lerobot-tagged[mujoco]"'
            ) from e

        if not self._frames:
            raise RuntimeError("No frames recorded. Call record() and stop_recording() first.")

        # Build or load the MuJoCo model
        if xml_path is not None:
            model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            xml = self._generate_scene_xml()
            model = mujoco.MjModel.from_xml_string(xml)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=height, width=width)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tags_by_id = self._detector.tags

        for frame in self._frames:
            # Position each detected object
            for loc in frame.tag_locations:
                tag = tags_by_id.get(loc.tag_id)
                body_name = f"object_{loc.tag_id}"

                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    continue

                jnt_adr = model.body_jntadr[body_id]
                if jnt_adr < 0:
                    continue

                x_mm, y_mm = loc.pos
                x = x_mm / 1000.0
                y = y_mm / 1000.0
                z = (tag.height_mm or 25) / 2000.0  # rest on floor

                qpos_adr = model.jnt_qposadr[jnt_adr]
                data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        renderer.close()
        writer.release()

    # ── internals ────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            ret, frame_img = self._cap.read()
            if not ret:
                break

            timestamp = time.monotonic() - t0
            tag_locations = self._detector.find(frame_img)

            self._frames.append(Frame(
                timestamp=timestamp,
                tag_locations=tag_locations,
                joint_positions=dict(_DUMMY_JOINTS),
                end_effector=(0.0, 0.0, 0.0),
            ))

    def _generate_scene_xml(self) -> str:
        """Auto-generate a MuJoCo XML scene with one box body per tracked tag."""
        bodies = []
        for tag in self._detector.tags.values():
            w = (tag.width_mm  or 25) / 2000.0  # half-size in metres
            h = (tag.height_mm or 25) / 2000.0
            d = (tag.depth_mm  or 25) / 2000.0
            bodies.append(
                f'    <body name="object_{tag.id}" pos="0 0 {h}">\n'
                f'      <freejoint/>\n'
                f'      <geom type="box" size="{w} {d} {h}" rgba="0.8 0.2 0.2 1"/>\n'
                f'    </body>'
            )

        bodies_xml = "\n".join(bodies)
        return f"""<mujoco model="lerobot-tagged-scene">
  <option timestep="0.002"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.85 0.85 0.85 1"/>
    <camera name="topdown" pos="0 0 0.8" euler="0 0 0"/>
{bodies_xml}
  </worldbody>
</mujoco>"""
